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


def isnan(x):
    return x != x


def mean(ll, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(list(ll))
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
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
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


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


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
        crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                 noise_channel=args.use_noise_channel)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=16, collate_fn=cr_collate)

        # crdata_valid = CRDataset(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='valid', stride=8)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = DataLoader(crdata_valid, batch_size=batch_size, shuffle=True, num_workers=0)

    else:
        crdata_train = CRDatasetSM(data_root=args.data_dir, config_dict=config_dict, split='train',
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

    focal = FocalLoss()
    Lovasz = LovaszSoftMax()
    dice = DiceLoss()
    print("Building model ... (%s)" % model_cfg)
    if model_cfg['type'] == 'CDC':
        rodnet = RODNet(n_class_train).cuda()
        criterion = nn.MSELoss()
    elif model_cfg['type'] == 'HG':
        rodnet = RODNet(n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwI':
        rodnet = RODNet(n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    else:
        raise TypeError

    # Multi-GPUs for training
    rodnet = nn.DataParallel(rodnet)
    rodnet = rodnet.cuda()

    optimizer = optim.Adam(rodnet.parameters(), lr=lr)
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
                    else:
                        loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap = loss_cur
                    if model_cfg['dice_loss']:
                        dice_loss = dice(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += dice_loss
                    if model_cfg['lovasz']:
                        Lovasz_loss = Lovasz(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += Lovasz_loss
                loss_confmap.backward()
                optimizer.step()
            else:
                if model_cfg['loss'] == 'focal':
                    focal_loss = FocalLoss(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap = focal_loss
                else:
                    loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
                if model_cfg['dice_loss']:
                    dice_loss = dice(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap += dice_loss
                if model_cfg['lovasz']:
                    Lovasz_loss = LovaszSoftMax(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap += Lovasz_loss
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

        scheduler.step()

    print('Training Finished.')
