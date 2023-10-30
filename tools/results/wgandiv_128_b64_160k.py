dataset_type = 'UnconditionalImageDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(128, 128)),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='UnconditionalImageDataset',
        imgs_root='../data/strawberry/',
        pipeline=[
            dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
            dict(type='Resize', keys=['real_img'], scale=(128, 128)),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['real_img']),
            dict(
                type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
        ]),
    val=dict(
        type='UnconditionalImageDataset',
        imgs_root=None,
        pipeline=[
            dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
            dict(type='Resize', keys=['real_img'], scale=(128, 128)),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['real_img']),
            dict(
                type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
        ]))
model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(type='WGANGPGenerator', noise_size=128, out_scale=128),
    discriminator=dict(
        type='WGANGPDiscriminator',
        in_channel=3,
        in_scale=128,
        conv_module_cfg=dict(
            conv_cfg=None,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=dict(type='GN'),
            order=('conv', 'norm', 'act'))),
    gan_loss=dict(type='GANLoss', gan_type='wgan'),
    disc_auxiliary_loss=[
        dict(
            type='DivergenceLoss',
            loss_weight=1,
            norm_mode='HWC',
            k=2,
            p=6,
            data_info=dict(
                discriminator='disc',
                real_data='real_imgs',
                fake_data='fake_imgs'))
    ])
train_cfg = dict(disc_steps=5)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9)),
    discriminator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.9)))
checkpoint_config = dict(interval=1, by_epoch=False)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
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
find_unused_parameters = True
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
lr_config = None
total_iters = 160000
metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=500),
    swd16k=dict(type='SWD', num_images=500, image_shape=(3, 128, 128)))
work_dir = './results'
gpu_ids = [0]
