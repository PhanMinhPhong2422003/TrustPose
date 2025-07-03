#  python /workspace/mmpose/tools/train.py /workspace/work/configs/TrustPose.py
# export PYTHONPATH="/workspace/work:$PYTHONPATH"
ex_name = "FULL_newdropout_pool"
_base_ = ['/workspace/mmpose/configs/_base_/default_runtime.py']
import datetime
import wandb

custom_hooks = [
    dict(type='SetEpochInfoHook'),  # Hiển thị epoch trong log
]

vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',init_kwargs=dict(project='HRPOSE', name= ex_name))  # Gửi log/ảnh lên Weights & Biases

]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
work_dir = '/workspace/work/checkpoint/TrustPose/' + ex_name

train_cfg = dict(max_epochs=300, val_interval=10)
custom_imports = dict(
    imports=['custom'], # <<< Tên file  (không có .py)
     allow_failed_imports=False)
# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[170, 230],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TeacherTopdownPoseEstimator',
    teacher_config='/workspace/work/configs/teacher.py',
    # teacher_dropout_epoch=100,
    teacher_dropout_iter = 5,

    data_preprocessor=dict(
        type='ExlPoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        # type='OldHRUNet,
        type='HRNet1',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),

        init_cfg=dict(
            type='Normal',
            ),
        conv_trans=(256, 32, 32, 32),
        conv_teacher_trans=(256, 32, 32, 32),        
    ),
    head=dict(
        type='TeacherHeatmapHead',
        in_channels=32,
        out_channels=14,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),

        loss_feat=dict(
            type='SCAMLoss',

            dist_type='cosine',  
            loss_weight=0.05,      
            feature_norm=True,
            avg_pool=False,
        ),
        
        loss_output_kl_cfg=dict(
            type = 'AdaptiveKLDivergenceLoss',
            temperature=1,
            loss_weight=2               
        ),

        loss_output_mse_cfg=None,
        out_w=1,
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'ExlposeDataset1'
data_mode = 'topdown'
data_root = '/workspace/data/ExLPose/'

# pipelines
train_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='ScalingLLPair'),
    dict(type='GetBBoxCenterScalePair'),
    dict(type='RandomFlipPair', direction='horizontal'),
    dict(type='RandomHalfBodyPair'),
    dict(type='RandomBBoxTransformPair'),
    dict(type='TopdownAffinePair', input_size=codec['input_size']),
    dict(type='GenerateTargetPair', encoder=codec),
    dict(type='PackPoseInputsPair')
]
val_pipeline = [
    dict(type='LoadImagePair'),
    dict(type='ScalingLLPair'),
    dict(type='GetBBoxCenterScalePair'),
    dict(type='TopdownAffinePair', input_size=codec['input_size']),
    dict(type='PackPoseInputsPair')
]

# data loaders
train_dataloader = dict(
    batch_size=40,
    num_workers=16,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/ExLPose/ExLPose_train_LL.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=40,
    num_workers=16,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/ExLPose/ExLPose_test_LL-A.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader


val_evaluator = [    
    dict(
        type='CocoMetric',
        ann_file=data_root + 'annotations/ExLPose/ExLPose_test_LL-A.json',
        use_area=False),

]
test_evaluator = val_evaluator

log_config = dict(  
    interval=50,
    hooks=[
        dict(type='WandbLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
