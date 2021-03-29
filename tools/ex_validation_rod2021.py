# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: validation_rod2021.py
@time: 2021/01/31
"""
import sys
import os

from cruw import CRUW
from cruw.eval import evaluate_rod2021, evaluate_rod2021_APAR


def eval_rod2021_batch():
    epoch_start, epoch_end = 1, 30
    pkl_idx = list(range(epoch_start, epoch_end + 1))
    config_file = 'configs/my_config_rodnet_hg1_win16_lovasz_bs16_lr1e4_2020_2_11.py'
    checkpoint_name = 'rodnet-hg1-win16-wobg-lovasz_bs16_lr1e4_2020_2_11-20210211-103448'
    for i in pkl_idx:
        cmd = 'python tools/validation.py --config %s \
        --data_dir /nfs/volume-95-8/ROD_Challenge/RODNet/data/zixiang_split/  \
        --valid \
        --checkpoint checkpoints/%s/epoch_%02d_final.pkl' % (config_file, checkpoint_name, i)

        os.system(cmd)

        data_root = "/nfs/volume-95-8/ROD_Challenge/src_dataset"
        dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
        submit_dir = '/nfs/volume-95-8/tianwanxin/RODNet/valid_results/%s' % checkpoint_name
        truth_dir = '/nfs/volume-95-8/ROD_Challenge/RODNet/for_validation/gt_zixiang_split'
        AP, AR = evaluate_rod2021_APAR(submit_dir, truth_dir, dataset)
        # print('epoch: %d, AP: %.4f, AR: %.4f' % (i, AP, AR))
        with open('/nfs/volume-95-8/tianwanxin/RODNet/valid_res/%s/valid_res.txt' % checkpoint_name, 'a') as f:
            f.write('epoch: %d, AP: %.4f, AR: %.4f\n' % (i, AP, AR))


if __name__ == '__main__':
    # data_root = "/nfs/volume-95-8/ROD_Challenge/src_dataset"
    # dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    # submit_dir = '/nfs/volume-95-8/ROD_Challenge/RODNet/tools/valid_results/rodnet-hg1-win16-wobg-20210206-124028'
    # truth_dir = '/nfs/volume-95-8/ROD_Challenge/RODNet/for_validation/gt_zixiang_split'
    # ap, ar = evaluate_rod2021_APAR(submit_dir, truth_dir, dataset)
    # print(ap, ar)

    eval_rod2021_batch()
