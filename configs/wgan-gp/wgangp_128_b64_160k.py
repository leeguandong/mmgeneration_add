_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/models/wgan/wgangp_base.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=2,  # luban_banner:64
    train=dict(imgs_root='../data/strawberry/'))

checkpoint_config = dict(interval=1, by_epoch=False)  # luban_banner:1000
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])  # luban_banner:1000

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)  # luban_banner:1000
]

lr_config = None
total_iters = 160000

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=500),  # num_images随便选的，想选多少就选多少
    swd16k=dict(type='SWD', num_images=500, image_shape=(3, 128, 128))
)
