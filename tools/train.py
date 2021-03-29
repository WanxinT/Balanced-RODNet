import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable

from cruw import CRUW
from cruw.eval import evaluate_rod2021, evaluate_rod2021_APAR

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.CRDatasetSM import CRDatasetSM
from rodnet.datasets.CRDataLoader import CRDataLoader
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file
from rodnet.utils.visualization import visualize_train_img

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')
    parser.add_argument('--save_memory', action="store_true", help="use customized dataloader to save memory")
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    args = parser.parse_args()
    return args


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='mean',
                   avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred - target).abs().pow(gamma) * \
            (target <= 0.0).float()
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def balanced_varifocal_loss(pred,
                   target,
                   weight=None,
                   alpha=0.75,
                   gamma=2.0,
                   iou_weighted=True,
                   reduction='sum',
                   avg_factor=None):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to False.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    target = target.type_as(pred)
    N = (target > 0.0).float().sum()
    M = (target <= 0.0).float().sum()
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() / N + \
            alpha * (pred - target).abs().pow(gamma) * \
            (target <= 0.0).float() / M
    else:
        focal_weight = (target > 0.0).float() / N + \
            alpha * (pred - target).abs().pow(gamma) * \
            (target <= 0.0).float() / M
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class BalancedCELoss(nn.Module):
    def __init__(self,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='sum',
                 loss_weight=1.0):
        """
        Args:
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to False.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(BalancedCELoss, self).__init__()

        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # pred and target should be of the same size
        assert pred.size() == target.size()
        target = target.type_as(pred)
        N = (target > 0.0).float().sum()
        M = (target <= 0.0).float().sum()
        if self.iou_weighted:
            focal_weight = target * (target > 0.0).float() / N + \
                           self.alpha * (target <= 0.0).float() / M
        else:
            focal_weight = (target > 0.0).float() / N + \
                           self.alpha * (target <= 0.0).float() / M
        loss = F.binary_cross_entropy(
            pred, target, reduction='none') * focal_weight
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss


class VarifocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True,
                 reduction='mean',
                 loss_weight=1.0):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to False.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to False.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(VarifocalLoss, self).__init__()
        """
        assert use_sigmoid is True, \
            'Only sigmoid varifocal loss supported now.'
        """
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * varifocal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


def isnan(x):
    return x != x


def mean(ll, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(ll)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0)), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P, C] Tensor, ground truth labels
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)  # C = 3
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = labels[:, c]  # foreground for class c
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    for i in range(C - 1):
        losses[0] += losses[i + 1]
    return losses[0] / C


def flatten_probas(probas, labels):
    """
    Flattens predictions in the batch
    """
    B, C, T, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # B * T * H * W, C = P, C
    labels = labels.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # B * T * H * W, C = P, C
    return probas, labels


class LovaszSoftMax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftMax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)


def diceloss(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = mean(diceloss_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = diceloss_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def diceloss_flat(probas, labels, classes='present', smooth=1):
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    class_pred = probas.view(-1)
    labels = labels.view(-1)

    intersection = (class_pred * labels).sum()
    dice = (2. * intersection + smooth) / (class_pred.sum() + labels.sum() + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=0):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return diceloss(probas, labels, self.classes, self.per_image, self.ignore)


def _neg_loss(pred, gt):
  """
  Modified focal loss. Exactly the same as CornerNet.
  Runs faster and costs a little bit more memory
  Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  """
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)
  loss = 0
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, preds, targets):
        pos = targets > 0
        num_pos = pos.data.long().sum()
        loc_loss = F.smooth_l1_loss(preds, targets, size_average=False)
        loss = loc_loss / num_pos
        return loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        """Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,21]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        """Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        """
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
        t = t[:, 1:]
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #         mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        #         masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        #         masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        #         loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        ################################################################

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        print('cls_loss: %.3f' % (cls_loss.data[0] / num_pos))
        loss = cls_loss / num_pos
        return loss


def eval_rod2021_batch(config_file, checkpoint_name):
    epoch_start, epoch_end = 1, 20
    pkl_idx = list(range(epoch_start, epoch_end + 1))
    for i in pkl_idx:
        cmd = 'python tools/validation.py --config %s \
        --data_dir /nfs/volume-95-8/ROD_Challenge/RODNet/data/scene_ave  \
        --valid \
        --checkpoint checkpoints/%s/epoch_%02d_final.pkl' % (config_file, checkpoint_name, i)

        os.system(cmd)

        data_root = "/nfs/volume-95-8/ROD_Challenge/src_dataset"
        dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
        submit_dir = '/nfs/volume-95-8/tianwanxin/RODNet/valid_results/%s' % checkpoint_name
        truth_dir = '/nfs/volume-95-8/aocheng/RODNeto/data/scene_ave/for_valid'
        AP, AR = evaluate_rod2021_APAR(submit_dir, truth_dir, dataset)
        # print('epoch: %d, AP: %.4f, AR: %.4f' % (i, AP, AR))
        with open('/nfs/volume-95-8/tianwanxin/RODNet/valid_res/%s/valid_res.txt' % checkpoint_name, 'a') as f:
            f.write('epoch: %d, AP: %.4f, AR: %.4f\n' % (i, AP, AR))

"""
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
"""


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    # dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'])
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name='sensor_config_rod2021')
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']
    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC as RODNet
    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG as RODNet
    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI as RODNet
    else:
        raise NotImplementedError

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir

    # create / load models
    cp_path = None
    epoch_start = 0
    iter_start = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)
    else:
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)

    train_viz_path = os.path.join(model_dir, 'train_viz')
    if not os.path.exists(train_viz_path):
        os.makedirs(train_viz_path)

    writer = SummaryWriter(model_dir)
    save_config_dict = {
        'args': vars(args),
        'config_dict': config_dict,
    }
    config_json_name = os.path.join(model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
    with open(config_json_name, 'w') as fp:
        json.dump(save_config_dict, fp)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, 'w'):
        pass

    n_class = dataset.object_cfg.n_class
    n_epoch = config_dict['train_cfg']['n_epoch']
    batch_size = config_dict['train_cfg']['batch_size']
    lr = config_dict['train_cfg']['lr']
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    print("Building dataloader ... (Mode: %s)" % ("save_memory" if args.save_memory else "normal"))

    if not args.save_memory:
        crdata_train = CRDataset(data_dir=config_dict['dataset_cfg']['pkl_root'], dataset=dataset, config_dict=config_dict, split='train',
                                 noise_channel=args.use_noise_channel)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

        # crdata_valid = CRDataset(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='valid', stride=8)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = DataLoader(crdata_valid, batch_size=batch_size, shuffle=True, num_workers=0)

    else:
        crdata_train = CRDatasetSM(data_root=config_dict['dataset_cfg']['pkl_root'], config_dict=config_dict, split='train',
                                   noise_channel=args.use_noise_channel)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = CRDataLoader(crdata_train, shuffle=True, noise_channel=args.use_noise_channel)

        # crdata_valid = CRDatasetSM(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='train', stride=8, is_Memory_Limit=True)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = CRDataLoader(crdata_valid, batch_size=batch_size, shuffle=True)

    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class

    focal = FocalLoss(num_classes=3)
    Vari_Focal = VarifocalLoss()
    smooth_l1_loss = SmoothL1Loss()
    Lovasz = LovaszSoftMax()
    dice = DiceLoss()
    if model_cfg['loss'] == 'BalancedCE':
        balanced_CE = BalancedCELoss(alpha=model_cfg['alpha'], iou_weighted=model_cfg['iou_weighted'])

    print("Building model ... (%s)" % model_cfg)
    if model_cfg['type'] == 'CDC':
        rodnet = RODNet(n_class_train).cuda()
        # criterion = nn.MSELoss()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HG':
        rodnet = RODNet(n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwI':
        rodnet = RODNet(n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    else:
        raise TypeError
    optimizer = optim.Adam(rodnet.parameters(), lr=lr)
    if 'gamma' in config_dict['train_cfg']:
        scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=config_dict['train_cfg']['gamma'])
    else:
        scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=0.1)
    iter_count = 0
    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        if 'optimizer_state_dict' in checkpoint:
            rodnet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            iter_start = checkpoint['iter'] + 1
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
        else:
            rodnet.load_state_dict(checkpoint)

    # print training configurations
    print("Model name: %s" % model_name)
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))

    for epoch in range(epoch_start, n_epoch):

        tic_load = time.time()
        # if epoch == epoch_start:
        #     dataloader_start = iter_start
        # else:
        #     dataloader_start = 0

        for iter, data_dict in enumerate(dataloader):

            data = data_dict['radar_data']
            image_paths = data_dict['image_paths']
            confmap_gt = data_dict['anno']['confmaps']
            if 'threshold' in model_cfg:
                confmap_gt[confmap_gt < model_cfg['threshold']] = 0.0
            if not data_dict['status']:
                # in case load npy fail
                print("Warning: Loading NPY data failed! Skip this iteration")
                tic_load = time.time()
                continue

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            confmap_preds = rodnet(data.float().cuda())

            loss_confmap = 0
            if stacked_num is not None:
                for i in range(stacked_num):
                    if model_cfg['loss'] == 'focal':
                        focal_loss = focal(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap = focal_loss
                    elif model_cfg['loss'] == 'VariFocal':
                        loss_cur = Vari_Focal(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap = loss_cur
                    elif model_cfg['loss'] == 'BalancedCE':
                        loss_cur = balanced_CE(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap = loss_cur
                    else:
                        loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap = loss_cur
                    if model_cfg['dice_loss']:
                        dice_loss = dice(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += dice_loss
                    if model_cfg['lovasz']:
                        Lovasz_loss = Lovasz(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += Lovasz_loss
                    if model_cfg['smooth_l1_loss']:
                        Smooth_L1_loss = smooth_l1_loss(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += Smooth_L1_loss
                loss_confmap.backward()
                optimizer.step()
            else:
                if model_cfg['loss'] == 'VariFocal':
                    criterion = Vari_Focal()
                elif model_cfg['loss'] == 'BalancedCE':
                    criterion = balanced_CE()
                if model_cfg['lovasz'] and model_cfg['loss'] == 'focal':
                    loss_confmap = Lovasz(confmap_preds, confmap_gt.float().cuda()) + focal(confmap_preds,
                                                                                                       confmap_gt.float().cuda())
                elif not model_cfg['loss'] == 'focal' and model_cfg['lovasz']:
                    loss_confmap = Lovasz(confmap_preds, confmap_gt.float().cuda()) + criterion(confmap_preds,
                                                                                                       confmap_gt.float().cuda())
                elif model_cfg['loss'] == 'focal' and not model_cfg['lovasz']:
                    loss_confmap = focal(confmap_preds, confmap_gt.float().cuda())
                else:
                    loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
                loss_confmap.backward()
                optimizer.step()

            if iter % config_dict['train_cfg']['log_step'] == 0:
                # print statistics
                print('epoch %2d, iter %4d: loss: %.8f | load time: %.4f | backward time: %.4f' %
                      (epoch + 1, iter + 1, loss_confmap.item(), tic - tic_load, time.time() - tic))
                with open(train_log_name, 'a+') as f_log:
                    f_log.write('epoch %2d, iter %4d: loss: %.8f | load time: %.4f | backward time: %.4f\n' %
                                (epoch + 1, iter + 1, loss_confmap.item(), tic - tic_load, time.time() - tic))

                if stacked_num is not None:
                    writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
                    confmap_pred = confmap_preds[stacked_num - 1].cpu().detach().numpy()
                else:
                    writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
                    confmap_pred = confmap_preds.cpu().detach().numpy()
                if 'mnet_cfg' in model_cfg:
                    chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, 0, :, :], radar_configs['data_type'])
                else:
                    chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, :, :], radar_configs['data_type'])

                # draw train images
                fig_name = os.path.join(train_viz_path,
                                        '%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
                img_path = image_paths[0][0]
                visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                    confmap_pred[0, :n_class, 0, :, :],
                                    confmap_gt[0, :n_class, 0, :, :])

            if (iter + 1) % config_dict['train_cfg']['save_step'] == 0:
                # validate current model
                # print("validing current model ...")
                # validate()

                # save current model
                print("saving current model ...")
                status_dict = {
                    'model_name': model_name,
                    'epoch': epoch,
                    'iter': iter,
                    'model_state_dict': rodnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_confmap,
                    'iter_count': iter_count,
                }
                save_model_path = '%s/epoch_%02d_iter_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
                torch.save(status_dict, save_model_path)

            iter_count += 1
            tic_load = time.time()

        # save current model
        print("saving current epoch model ...")
        status_dict = {
            'model_name': model_name,
            'epoch': epoch,
            'iter': iter,
            'model_state_dict': rodnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_confmap,
            'iter_count': iter_count,
        }
        save_model_path = '%s/epoch_%02d_final.pkl' % (model_dir, epoch + 1)
        torch.save(status_dict, save_model_path)


        data_root = config_dict['dataset_cfg']['pkl_root']
        cmd = 'python /nfs/volume-95-8/tianwanxin/RODNet/tools/validation.py --config /nfs/volume-95-8/tianwanxin/RODNet/%s \
        --data_dir %s  \
        --valid \
        --checkpoint /nfs/volume-95-8/tianwanxin/RODNet/checkpoints/%s/epoch_%02d_final.pkl' % (args.config, data_root, model_name, epoch + 1)

        os.system(cmd)

        dataset = CRUW(data_root="/nfs/volume-95-8/ROD_Challenge/src_dataset", sensor_config_name='sensor_config_rod2021')
        submit_dir = '/nfs/volume-95-8/tianwanxin/RODNet/valid_results/%s' % model_name
        # truth_dir = '/nfs/volume-95-8/aocheng/RODNeto/data/scene_ave/for_valid'
        truth_dir = config_dict['dataset_cfg']['val_root']
        AP, AR = evaluate_rod2021_APAR(submit_dir, truth_dir, dataset)

        print('epoch: %d, AP: %.4f, AR: %.4f' % (epoch, AP, AR))
        valid_res_dir = '/nfs/volume-95-8/tianwanxin/RODNet/valid_res/'
        res_dir = os.path.join(valid_res_dir, model_name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        with open('/nfs/volume-95-8/tianwanxin/RODNet/valid_res/%s/valid_res.txt' % model_name, 'a') as f:
            f.write('epoch: %d, AP: %.4f, AR: %.4f\n' % (epoch + 1, AP, AR))

        scheduler.step()

    print('Training Finished.')