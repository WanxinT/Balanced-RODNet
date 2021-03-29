# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: instructions.py
@time: 2021/01/25
"""

# Prepare data for ROD2021 Challenge
# with labels
python tools/prepare_dataset/prepare_data.py \
        --config configs/my_config_rodnet_cdc_win16.py \
        --data_root /nfs/volume-95-8/ROD_Challenge/src_dataset \
        --split train \
        --out_data_dir data/rod2021

# without labels
python tools/prepare_dataset/prepare_data_demo.py \
        --config configs/my_config_rodnet_cdc_win16.py \
        --data_root /nfs/volume-95-8/ROD_Challenge/src_dataset \
        --split demo \
        --out_data_dir data/rod2021

# Train models
nohup python tools/train.py --config configs/my_config_rodnet_cdc_win16.py \
        --data_dir data/rod2021 \
        --log_dir checkpoints/ &

nohup python tools/train_multigpu.py --config configs/my_config_rodnet_cdc_win16.py \
        --data_dir data/rod2021 \
        --log_dir checkpoints/ &

# Test models multi-gpus
python tools/test.py --config configs/my_config_rodnet_cdc_win16.py \
        --data_dir data/rod2021 \
        --checkpoint checkpoints/rodnet-cdc-win16-wobg-20210128-124943/epoch_100_final.pkl
