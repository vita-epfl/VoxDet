data_root = '/mnt/vita/scratch/datasets/SemanticKITTI/dataset/'
ann_file = '/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/'
stereo_depth_root = '/mnt/vita/scratch/datasets/SemanticKITTI/depth/'
camera_used = ['left']
dataset_type = 'SemanticKITTIDatasetLC'
point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
lss_downsample = [2, 2, 2]
voxel_x = 0.2
voxel_y = 0.2
voxel_z = 0.2
grid_config = dict(
    xbound=[0, 51.2, 0.4],
    ybound=[-25.6, 25.6, 0.4],
    zbound=[-2, 4.4, 0.4],
    dbound=[2.0, 58.0, 0.5])
empty_idx = 0
semantic_kitti_class_frequencies = [
    5417730330.0, 15783539.0, 125136.0, 118809.0, 646799.0, 821951.0, 262978.0,
    283696.0, 204750.0, 61688703.0, 4502961.0, 44883650.0, 2269923.0,
    56840218.0, 15719652.0, 158442623.0, 2061623.0, 36970522.0, 1151988.0,
    334146.0
]
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign'
]
num_class = 20
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0)
data_config = dict(
    input_size=(384, 1280),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_size = [128, 128, 16]
coarse_ratio = 2
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=True,
        color_jitter=(0.4, 0.4, 0.4)),
    dict(
        type='LoadLidarPointsFromFiles_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        is_train=True),
    dict(
        type='LidarPointsPreProcess_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        occ_size=[256, 256, 32],
        coarse_ratio=2,
        is_train=True),
    dict(
        type='CreateDepthFromLiDAR',
        data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
        dataset='kitti'),
    dict(
        type='LoadSemKittiAnnotation',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=True,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'],
        meta_keys=['pc_range', 'occ_size', 'gt_occ_1_2'])
]
trainset_config = dict(
    type='SemanticKITTIDatasetLC',
    stereo_depth_root='/mnt/vita/scratch/datasets/SemanticKITTI/depth/',
    data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
    ann_file='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=True,
            color_jitter=(0.4, 0.4, 0.4)),
        dict(
            type='LoadLidarPointsFromFiles_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            is_train=True),
        dict(
            type='LidarPointsPreProcess_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
            occ_size=[256, 256, 32],
            coarse_ratio=2,
            is_train=True),
        dict(
            type='CreateDepthFromLiDAR',
            data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
            dataset='kitti'),
        dict(
            type='LoadSemKittiAnnotation',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=True,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=[
                'points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'
            ],
            meta_keys=['pc_range', 'occ_size', 'gt_occ_1_2'])
    ],
    split='train',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
    test_mode=False)
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        load_stereo_depth=True,
        is_train=False,
        color_jitter=None),
    dict(
        type='LoadLidarPointsFromFiles_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        is_train=False),
    dict(
        type='LidarPointsPreProcess_SemanticKitti',
        data_config=dict(
            input_size=(384, 1280),
            resize=(0.0, 0.0),
            rot=(0.0, 0.0),
            flip=False,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        occ_size=[256, 256, 32],
        coarse_ratio=2,
        is_train=False),
    dict(
        type='CreateDepthFromLiDAR',
        data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
        dataset='kitti'),
    dict(
        type='LoadSemKittiAnnotation',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            flip_dz_ratio=0),
        apply_bda=False,
        is_train=False,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    dict(
        type='CollectData',
        keys=['points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'],
        meta_keys=[
            'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
            'stereo_depth'
        ])
]
testset_config = dict(
    type='SemanticKITTIDatasetLC',
    stereo_depth_root='/mnt/vita/scratch/datasets/SemanticKITTI/depth/',
    data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
    ann_file='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/',
    pipeline=[
        dict(
            type='LoadMultiViewImageFromFiles_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            load_stereo_depth=True,
            is_train=False,
            color_jitter=None),
        dict(
            type='LoadLidarPointsFromFiles_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            is_train=False),
        dict(
            type='LidarPointsPreProcess_SemanticKitti',
            data_config=dict(
                input_size=(384, 1280),
                resize=(0.0, 0.0),
                rot=(0.0, 0.0),
                flip=False,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
            occ_size=[256, 256, 32],
            coarse_ratio=2,
            is_train=False),
        dict(
            type='CreateDepthFromLiDAR',
            data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
            dataset='kitti'),
        dict(
            type='LoadSemKittiAnnotation',
            bda_aug_conf=dict(
                rot_lim=(-22.5, 22.5),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5,
                flip_dz_ratio=0),
            apply_bda=False,
            is_train=False,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
        dict(
            type='CollectData',
            keys=[
                'points', 'grid_ind', 'voxel_position_grid_coarse', 'gt_occ'
            ],
            meta_keys=[
                'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                'stereo_depth'
            ])
    ],
    split='test',
    camera_used=['left'],
    occ_size=[256, 256, 32],
    pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4])
data = dict(
    train=dict(
        type='SemanticKITTIDatasetLC',
        stereo_depth_root='/mnt/vita/scratch/datasets/SemanticKITTI/depth/',
        data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
        ann_file='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=True,
                color_jitter=(0.4, 0.4, 0.4)),
            dict(
                type='LoadLidarPointsFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                is_train=True),
            dict(
                type='LidarPointsPreProcess_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                occ_size=[256, 256, 32],
                coarse_ratio=2,
                is_train=True),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
                dataset='kitti'),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=True,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=[
                    'points', 'grid_ind', 'voxel_position_grid_coarse',
                    'gt_occ'
                ],
                meta_keys=['pc_range', 'occ_size', 'gt_occ_1_2'])
        ],
        split='train',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        test_mode=False),
    val=dict(
        type='SemanticKITTIDatasetLC',
        stereo_depth_root='/mnt/vita/scratch/datasets/SemanticKITTI/depth/',
        data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
        ann_file='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='LoadLidarPointsFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                is_train=False),
            dict(
                type='LidarPointsPreProcess_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                occ_size=[256, 256, 32],
                coarse_ratio=2,
                is_train=False),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
                dataset='kitti'),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=[
                    'points', 'grid_ind', 'voxel_position_grid_coarse',
                    'gt_occ'
                ],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth'
                ])
        ],
        split='test',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
    test=dict(
        type='SemanticKITTIDatasetLC',
        stereo_depth_root='/mnt/vita/scratch/datasets/SemanticKITTI/depth/',
        data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
        ann_file='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/labels/',
        pipeline=[
            dict(
                type='LoadMultiViewImageFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                load_stereo_depth=True,
                is_train=False,
                color_jitter=None),
            dict(
                type='LoadLidarPointsFromFiles_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                is_train=False),
            dict(
                type='LidarPointsPreProcess_SemanticKitti',
                data_config=dict(
                    input_size=(384, 1280),
                    resize=(0.0, 0.0),
                    rot=(0.0, 0.0),
                    flip=False,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                occ_size=[256, 256, 32],
                coarse_ratio=2,
                is_train=False),
            dict(
                type='CreateDepthFromLiDAR',
                data_root='/mnt/vita/scratch/datasets/SemanticKITTI/dataset/',
                dataset='kitti'),
            dict(
                type='LoadSemKittiAnnotation',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                    flip_dz_ratio=0),
                apply_bda=False,
                is_train=False,
                point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]),
            dict(
                type='CollectData',
                keys=[
                    'points', 'grid_ind', 'voxel_position_grid_coarse',
                    'gt_occ'
                ],
                meta_keys=[
                    'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img',
                    'stereo_depth'
                ])
        ],
        split='test',
        camera_used=['left'],
        occ_size=[256, 256, 32],
        pc_range=[0, -25.6, -2, 51.2, 25.6, 4.4]))
train_dataloader_config = dict(batch_size=1, num_workers=4)
test_dataloader_config = dict(batch_size=1, num_workers=4)
_dim_ = 128
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='VoxDetLiDAR',
    car_scale_filter_max=[30, 30, 30],
    use_gt_refine=True,
    lidar_tokenizer=dict(
        type='LidarEncoder',
        grid_size=[128, 128, 16],
        in_channels=6,
        out_channels=128,
        fea_compre=None,
        base_channels=128,
        split=[8, 8, 8],
        track_running_stats=True),
    lidar_backbone=dict(
        type='CustomResNet3D',
        numC_input=128,
        num_layer=[2, 2, 2],
        drop_path_rate=0.3,
        num_channels=[128, 128, 128],
        stride=[1, 2, 2]),
    lidar_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[128, 128, 128],
        out_channels=128,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='Conv3d'),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='trilinear', align_corners=False)),
    occ_encoder_backbone=dict(
        type='Ident',
        embed_dims=128,
        local_aggregator=dict(
            type='VoxelAggregatorDual',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[3, 3, 3],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2],
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                drop_path_rate=0.3),
            local_encoder_neck=dict(
                type='SpatiallyDecoupledFPN',
                share_fpn=False,
                in_channels=[128, 128, 128],
                out_channels=128,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear', align_corners=False)))),
    pts_bbox_head_aux=dict(
        type='OccHead',
        in_channels=[128],
        out_channel=20,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=0.5,
            loss_voxel_sem_scal_weight=0.5,
            loss_voxel_geo_scal_weight=0.5),
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=[
            5417730330.0, 15783539.0, 125136.0, 118809.0, 646799.0, 821951.0,
            262978.0, 283696.0, 204750.0, 61688703.0, 4502961.0, 44883650.0,
            2269923.0, 56840218.0, 15719652.0, 158442623.0, 2061623.0,
            36970522.0, 1151988.0, 334146.0
        ]),
    pts_bbox_head=dict(
        type='VoxDetHead',
        down_sampling_ratio=0.5,
        balance_reg_loss='none',
        box_down_sample='trilinear',
        align_corners=False,
        num_inst_layer=4,
        use_bias=False,
        isolation_scale=0,
        pred_six_directions=True,
        in_channels=[128],
        out_channel=20,
        empty_idx=0,
        num_level=1,
        with_cp=False,
        occ_size=[256, 256, 32],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=3.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=5.0,
            loss_voxel_ctr_weight=1.0),
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=[
            5417730330.0, 15783539.0, 125136.0, 118809.0, 646799.0, 821951.0,
            262978.0, 283696.0, 204750.0, 61688703.0, 4502961.0, 44883650.0,
            2269923.0, 56840218.0, 15719652.0, 158442623.0, 2061623.0,
            36970522.0, 1151988.0, 334146.0
        ]))
learning_rate = 0.0003
training_steps = 25000
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.01)
sync_bn = True
lr_scheduler = dict(
    type='OneCycleLR',
    max_lr=0.0003,
    total_steps=25010,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy='cos',
    interval='step',
    frequency=1)
config_path = 'configs/voxdet-semnatickitti-r50-lidar-v3.py'
ckpt_path = None
seed = 42
log_folder = 'exps/voxdet-semnatickitti-r50-lidar-v3'
save_path = None
test_mapping = False
submit = False
eval = False
log_every_n_steps = 100
check_val_every_n_epoch = 1
pretrain = False
