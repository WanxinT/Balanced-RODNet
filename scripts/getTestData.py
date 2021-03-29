# -*- coding:utf-8 -*-
"""
@author:Zehui Yu
@file: getTestData.py
@time: 2021/01/25
"""
import os
import random
import shutil

if __name__ == '__main__':
    total_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/data/rod2021/total'
    totalNum = len(os.listdir(total_root))
    print("total Num: %d." % totalNum)

    testRate = 0.1
    testNum = int(totalNum * testRate)
    print("test Num: %d." % testNum)

    testIdx = random.sample(list(range(totalNum)), testNum)

    train_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/data/rod2021/train'
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    else:
        print("delete existed test root.")
        shutil.rmtree(train_root)
        os.makedirs(train_root)

    test_root = '/nfs/volume-95-8/ROD_Challenge/RODNet/data/rod2021/test'
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    else:
        print("delete existed test root.")
        shutil.rmtree(test_root)
        os.makedirs(test_root)

    for i, name in enumerate(sorted(os.listdir(total_root))):
        if i in testIdx:
            shutil.copy(os.path.join(total_root, name), os.path.join(test_root, name))
        else:
            shutil.copy(os.path.join(total_root, name), os.path.join(train_root, name))

    print("Done.")