# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: dataStatistic.py
@time: 2021/02/01
"""
import os


def ClassName_statistic(anno_root):
    className_num_dict = {
        'cyclist': 0,
        'pedestrian': 0,
        'car': 0,
    }
    for anno_txt in os.listdir(anno_root):
        with open(os.path.join(anno_root, anno_txt), 'r') as f:
            data = f.readlines()

        for line in data:
            line = line.strip().split(' ')
            className = line[-1]
            assert className in className_num_dict
            className_num_dict[className] += 1

    print(className_num_dict)


if __name__ == '__main__':
    ClassName_statistic(anno_root='/nfs/volume-95-8/ROD_Challenge/src_dataset/TRAIN_RAD_H_ANNO')
