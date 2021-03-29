import os
import sys
sys.path.append('/nfs/volume-95-8/aocheng/RODNeto/')
import time
import argparse
import numpy as np
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cruw import CRUW

from rodneto.datasets.CRDataset import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.post_processing import post_process, post_process_single_frame
from rodneto.core.post_processing import write_dets_results, write_dets_results_single_frame
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from rodnet.utils.load_configs import load_configs_from_file
from rodnet.utils.solve_dir import create_random_model_name

from cruw import CRUW
from cruw.eval import evaluate_rod2021, evaluate_rod2021_APAR

"""
Example:
    python test.py -m HG -dd /mnt/ssd2/rodnet/data/ -ld /mnt/ssd2/rodnet/checkpoints/ \
        -md HG-20200122-104604 -rd /mnt/ssd2/rodnet/results/
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')
    parser.add_argument('--config', type=str, help='choose rodnet model configurations')
    parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--checkpoint', type=str, help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default='./results/', help='directory to save testing results')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')
    args = parser.parse_args()
    return args


def validate4(args, rodnet, model_name, res_dir):
    # args = parse_args()
    # sybl = args.symbol

    config_dict = load_configs_from_file(args.config)
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name='sensor_config_rod2021')
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_configs = config_dict['model_cfg']

    # parameter settings
    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']

    win_size = train_configs['win_size']
    n_class = dataset.object_cfg.n_class

    if 'stacked_num' in model_configs:
        stacked_num = model_configs['stacked_num']
    else:
        stacked_num = None

    confmap_shape = (n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

    if args.use_noise_channel:
        n_class_test = n_class + 1
    else:
        n_class_test = n_class

    rodnet.eval()

    test_res_dir = os.path.join(os.path.join(res_dir, '%s_val_tmp' % model_name))   # tmp dir for save txt during validation, to output AP and AR
    if not os.path.exists(test_res_dir):
        os.makedirs(test_res_dir)

    total_time = 0
    total_count = 0

    data_root = dataset_configs['pkl_root']   # edited
    seq_names = sorted(os.listdir(os.path.join(data_root, 'valid')))
    print(seq_names)

    for seq_name in seq_names:
        # seq_name = seq_name.split('.')[0]
        # seq_res_dir = os.path.join(test_res_dir, seq_name)
        # if not os.path.exists(seq_res_dir):
        #     os.makedirs(seq_res_dir)
        # seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
        # if not os.path.exists(seq_res_viz_dir):
        #     os.makedirs(seq_res_viz_dir)
        f = open(os.path.join(test_res_dir, '%s.txt' % seq_name.split('.')[0]), 'w')
        f.close()

    for subset in seq_names:
        print(subset)
        crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='valid',
                                noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False)
        print("Length of testing data: %d" % len(crdata_test))
        dataloader = DataLoader(crdata_test, batch_size=4, shuffle=False, num_workers=16, collate_fn=cr_collate)

        seq_names = crdata_test.seq_names
        index_mapping = crdata_test.index_mapping

        init_genConfmap = ConfmapStack(confmap_shape)
        iter_ = init_genConfmap
        for i in range(train_configs['win_size'] - 1):
            while iter_.next is not None:
                iter_ = iter_.next
            iter_.next = ConfmapStack(confmap_shape)

        load_tic = time.time()
        for iter, data_dict in enumerate(dataloader):
            load_time = time.time() - load_tic
            data = data_dict['radar_data']
            image_paths = data_dict['image_paths'][0]
            seq_name = data_dict['seq_names'][0]

            confmap_gt = data_dict['anno']['confmaps']
            obj_info = data_dict['anno']['obj_infos']

            save_path = os.path.join(test_res_dir, '%s.txt' % seq_name.split('.')[0])
            start_frame_name = image_paths[0].split('/')[-1].split('.')[0]
            end_frame_name = image_paths[-1].split('/')[-1].split('.')[0]
            start_frame_id = int(start_frame_name)
            end_frame_id = int(end_frame_name)

            if iter%20==0:
                print("Testing %s: %s-%s" % (seq_name, start_frame_name, end_frame_name))
            tic = time.time()
            with torch.no_grad():
                imagData=torch.cat([data,data],axis=1)
                imagData[:,2,:,:,:]=torch.sqrt(data[:,0,:,:,:]**2+data[:,1,:,:,:]**2)
                imagData[:,3,:,:,:]=torch.atan2(data[:,0,:,:,:],data[:,1,:,:,:])
            confmap_pred = rodnet(imagData.float().cuda())
            if stacked_num is not None:
                confmap_pred = confmap_pred[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
            else:
                confmap_pred = confmap_pred.cpu().detach().numpy()

            if args.use_noise_channel:
                confmap_pred = confmap_pred[:, :n_class, :, :, :]

            infer_time = time.time() - tic
            total_time += infer_time

            iter_ = init_genConfmap
            for i in range(confmap_pred.shape[2]):
                if iter_.next is None and i != confmap_pred.shape[2] - 1:
                    iter_.next = ConfmapStack(confmap_shape)
                iter_.append(confmap_pred[0, :, i, :, :])
                iter_ = iter_.next

            process_tic = time.time()
            for i in range(test_configs['test_stride']):
                total_count += 1
                res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                cur_frame_id = start_frame_id + i
                write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                # confmap_pred_0 = init_genConfmap.confmap
                # res_final_0 = res_final
                # img_path = image_paths[i]
                # radar_input = chirp_amp(data.numpy()[0, :, i, :, :], radar_configs['data_type'])
                # fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                # if confmap_gt is not None:
                #     confmap_gt_0 = confmap_gt[0, :, i, :, :]
                #     visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0,
                #                        dataset, sybl=sybl)
                # else:
                #     visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
                #                              dataset, sybl=sybl)
                init_genConfmap = init_genConfmap.next

            if iter == len(dataloader) - 1:
                offset = test_configs['test_stride']
                cur_frame_id = start_frame_id + offset
                while init_genConfmap is not None:
                    total_count += 1
                    res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                    write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                    # confmap_pred_0 = init_genConfmap.confmap
                    # res_final_0 = res_final
                    # img_path = image_paths[offset]
                    # radar_input = chirp_amp(data.numpy()[0, :, offset, :, :], radar_configs['data_type'])
                    # fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                    # if confmap_gt is not None:
                    #     confmap_gt_0 = confmap_gt[0, :, offset, :, :]
                    #     visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0,
                    #                        dataset, sybl=sybl)
                    # else:
                    #     visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
                    #                              dataset, sybl=sybl)
                    init_genConfmap = init_genConfmap.next
                    offset += 1
                    cur_frame_id += 1

            if init_genConfmap is None:
                init_genConfmap = ConfmapStack(confmap_shape)

            proc_time = time.time() - process_tic
            if iter%20==0:
                print("Load time: %.4f | Inference time: %.4f | Process time: %.4f" % (load_time, infer_time, proc_time))

            load_tic = time.time()

    print("ave time: %f" % (total_time / total_count))

    # data_root = "/nfs/volume-95-8/ROD_Challenge/src_dataset"
    # dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    truth_dir = '/nfs/volume-95-8/aocheng/RODNeto/data/fold2/for_valid'

    print(test_res_dir,truth_dir,dataset)
    AP, AR = evaluate_rod2021_APAR(test_res_dir, truth_dir, dataset)
    shutil.rmtree(test_res_dir)
    print(AP,AR)

    rodnet.train()

    return AP, AR
