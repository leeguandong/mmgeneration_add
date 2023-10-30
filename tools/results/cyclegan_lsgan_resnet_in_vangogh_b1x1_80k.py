_domain_a = None
_domain_b = None
model = dict(
    type='CycleGAN',
    generator=dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain='photo',
    reachable_domains=['vangogh', 'photo'],
    related_domains=['vangogh', 'photo'],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(pred='cycle_vangogh', target='real_vangogh'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(pred='cycle_photo', target='real_photo'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(pred='identity_vangogh', target='real_vangogh'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(pred='identity_photo', target='real_photo'),
            reduction='mean')
    ])
train_cfg = dict(buffer_size=50)
test_cfg = None
train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
domain_a = 'vangogh'
domain_b = 'photo'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_vangogh',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_photo',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_vangogh', 'img_photo'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=['img_vangogh', 'img_photo'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=['img_vangogh'], direction='horizontal'),
    dict(type='Flip', keys=['img_photo'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_vangogh', 'img_photo']),
    dict(
        type='Normalize',
        keys=['img_vangogh', 'img_photo'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_vangogh', 'img_photo']),
    dict(
        type='Collect',
        keys=['img_vangogh', 'img_photo'],
        meta_keys=['img_vangogh_path', 'img_photo_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_vangogh',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_photo',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_vangogh', 'img_photo'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_vangogh', 'img_photo']),
    dict(
        type='Normalize',
        keys=['img_vangogh', 'img_photo'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_vangogh', 'img_photo']),
    dict(
        type='Collect',
        keys=['img_vangogh', 'img_photo'],
        meta_keys=['img_vangogh_path', 'img_photo_path'])
]
data_root = None
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type='UnpairedImageDataset',
        dataroot='../data/vangogh2photo/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_vangogh',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_photo',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_vangogh', 'img_photo'],
                scale=(286, 286),
                interpolation='bicubic'),
            dict(
                type='Crop',
                keys=['img_vangogh', 'img_photo'],
                crop_size=(256, 256),
                random_crop=True),
            dict(type='Flip', keys=['img_vangogh'], direction='horizontal'),
            dict(type='Flip', keys=['img_photo'], direction='horizontal'),
            dict(type='RescaleToZeroOne', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Normalize',
                keys=['img_vangogh', 'img_photo'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Collect',
                keys=['img_vangogh', 'img_photo'],
                meta_keys=['img_vangogh_path', 'img_photo_path'])
        ],
        test_mode=False,
        domain_a='vangogh',
        domain_b='photo'),
    val=dict(
        type='UnpairedImageDataset',
        dataroot='../data/vangogh2photo/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_vangogh',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_photo',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_vangogh', 'img_photo'],
                scale=(256, 256),
                interpolation='bicubic'),
            dict(type='RescaleToZeroOne', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Normalize',
                keys=['img_vangogh', 'img_photo'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Collect',
                keys=['img_vangogh', 'img_photo'],
                meta_keys=['img_vangogh_path', 'img_photo_path'])
        ],
        test_mode=True,
        domain_a='vangogh',
        domain_b='photo'),
    test=dict(
        type='UnpairedImageDataset',
        dataroot='../data/vangogh2photo/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_vangogh',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_photo',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_vangogh', 'img_photo'],
                scale=(256, 256),
                interpolation='bicubic'),
            dict(type='RescaleToZeroOne', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Normalize',
                keys=['img_vangogh', 'img_photo'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_vangogh', 'img_photo']),
            dict(
                type='Collect',
                keys=['img_vangogh', 'img_photo'],
                meta_keys=['img_vangogh_path', 'img_photo_path'])
        ],
        test_mode=True,
        domain_a='vangogh',
        domain_b='photo'))
checkpoint_config = dict(interval=1000, by_epoch=False, save_optimizer=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_vangogh', 'fake_photo'],
        interval=2500)
]
runner = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
dataroot = '../data/vangogh2photo/'
optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
total_iters = 80000
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=40000, interval=400)
use_ddp_wrapper = True
num_images = 106
metrics = dict(
    FID=dict(type='FID', num_images=106, image_shape=(3, 256, 256)),
    IS=dict(
        type='IS',
        num_images=106,
        image_shape=(3, 256, 256),
        inception_args=dict(type='pytorch')))
evaluation = dict(
    type='TranslationEvalHook',
    target_domain='photo',
    interval=1000,
    metrics=[
        dict(type='FID', num_images=106, bgr2rgb=True),
        dict(type='IS', num_images=106, inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])
work_dir = './results'
gpu_ids = [0]
