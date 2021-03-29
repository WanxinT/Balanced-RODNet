# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: move2validation.py
@time: 2021/01/29
"""
import os
import shutil

results_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/results'
model_name = 'rodnet-cdc-win16-wobg-20210128-124943'
valid_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/for_validation/submit'
gt_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/for_validation/gt'
valid_seq_names = []

# for submit txt
for dir in os.listdir(os.path.join(results_root, model_name)):
    if dir.endswith('txt'):
        continue
    seq_name = dir
    valid_seq_names.append(seq_name)
    seq_txt_path = os.path.join(results_root, model_name, seq_name, 'rod_res.txt')
    tgt_txt_path = os.path.join(valid_root, '%s.txt' % seq_name)
    shutil.copy(seq_txt_path, tgt_txt_path)

# for ground truth txt
gt_txt_root = '/nfs/volume-95-8/ROD_Challenge/src_dataset/TRAIN_RAD_H_ANNO'
for seq_name in valid_seq_names:
    gt_src = os.path.join(gt_txt_root, '%s.txt' % seq_name)
    gt_tgt = os.path.join(gt_root, '%s.txt' % seq_name)
    shutil.copy(gt_src, gt_tgt)