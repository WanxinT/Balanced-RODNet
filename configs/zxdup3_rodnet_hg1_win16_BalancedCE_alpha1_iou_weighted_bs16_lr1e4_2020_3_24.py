dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="/nfs/volume-95-8/ROD_Challenge/src_dataset",
    data_root="/nfs/volume-95-8/ROD_Challenge/src_dataset/TRAIN_CAM_0",
    radar_root="/nfs/volume-95-8/ROD_Challenge/src_dataset/TRAIN_RAD_H",
    anno_root="/nfs/volume-95-8/ROD_Challenge/src_dataset/TRAIN_RAD_H_ANNO",
    pkl_root="/nfs/volume-95-8/aocheng/RODNeto/data/zx_dup3",
    val_root="/nfs/volume-95-8/aocheng/RODNeto/data/zx_dup3/for_valid",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        # seqs=[],  # can choose from the subdir folder
    ),
    valid=dict(
        subdir='valid',
        # seqs=[],
    ),
    test=dict(
        subdir='test',
        # seqs=[],  # can choose from the subdir folder
    ),
    demo=dict(
        subdir='demo',
        # seqs=[],
    ),
)

model_cfg = dict(
    type='HG',
    name='zx_dup3_rodnet-hg1-win16-wobg_BalancedCE_alpha1_iou_weighted_bs16_lr1e4_2020_3_24',
    max_dets=20,
    peak_thres=0.4,
    ols_thres=0.3,
    stacked_num=1,
    loss='BalancedCE',   # 'VariFocal' 'focal' 'BCE'
    alpha=1.0,
    iou_weighted=True,
    lovasz=False,
    dice_loss=False,  
    smooth_l1_loss=False, 
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)

train_cfg = dict(
    n_epoch=20,
    batch_size=16,
    lr=0.0001,
    gamma=0.1,
    lr_step=5,  # lr will decrease 10 times after lr_step epoches
    win_size=16,
    train_step=1,
    train_stride=4,
    log_step=100,
    save_step=1000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
