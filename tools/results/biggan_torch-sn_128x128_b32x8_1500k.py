model = dict(
    type='BasiccGAN',
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=4,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-06,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False,
        sn_style='torch'),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=4,
        base_channels=96,
        sn_eps=1e-06,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True,
        sn_style='torch'),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))
train_cfg = dict(
    disc_steps=1, gen_steps=1, batch_accumulation_steps=8, use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-06),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-06))
dataset_type = 'mmcls.CustomDataset'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='mmcls.CustomDataset',
        data_prefix='F:/Dataset/styledata/bgv2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CenterCropLongEdge', keys=['img']),
            dict(type='Resize', size=(128, 128), backend='pillow'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='mmcls.CustomDataset',
        data_prefix='F:/Dataset/styledata/bgv2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CenterCropLongEdge', keys=['img']),
            dict(type='Resize', size=(128, 128), backend='pillow'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='mmcls.CustomDataset',
        data_prefix='F:/Dataset/styledata/bgv2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CenterCropLongEdge', keys=['img']),
            dict(type='Resize', size=(128, 128), backend='pillow'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=8,
        start_iter=160000,
        interp_cfg=dict(momentum=0.9999, momentum_nontrainable=0.9999),
        priority='VERY_HIGH')
]
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 10000)]
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
lr_config = None
total_iters = 1500000
use_ddp_wrapper = True
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(type='FID', num_images=500, bgr2rgb=True),
        dict(type='IS', num_images=500)
    ],
    sample_kwargs=dict(sample_model='ema'),
    best_metric=['fid', 'is'])
metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=500,
        bgr2rgb=True,
        inception_args=dict(type='StyleGAN')),
    is50k=dict(type='IS', num_images=500))
work_dir = './results'
gpu_ids = [0]
