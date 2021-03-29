# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: validation_rod2021.py
@time: 2021/01/31
"""

import sys

from cruw import CRUW
from cruw.eval import evaluate_rod2021

if __name__ == '__main__':
    data_root = "/nfs/volume-95-8/ROD_Challenge/src_dataset"
    dataset = CRUW(data_root=data_root, sensor_config_name='sensor_config_rod2021')
    submit_dir = '/nfs/volume-95-8/tianwanxin/RODNet/valid_results/rodnet-hg1-win16-wobg_VariFocal_bs16_lr1e4_2020_3_19-20210323-130242'
    truth_dir = '/nfs/volume-95-8/aocheng/RODNeto/data/fold2-CS-HW/for_valid'
    evaluate_rod2021(submit_dir, truth_dir, dataset)
