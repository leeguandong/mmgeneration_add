_base_ = [
    '../_base_/models/layoutgan/layoutnetpp.py', '../_base_/datasets/magazine.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=500, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

total_iters = 30000

custom_hooks = [
    dict(
        type='MMGenVisualizationHook',  # 这种写法更好一点
        output_dir='visual',
        interval=1,
        rerange=False,
        bgr2rgb=False,
        res_name_list=['fake_imgs', 'real_imgs']),
]

evaluation = dict(
    type='LayoutEvalHook',
    interval=1,
    metrics=dict(
        type='FID',
        num_images=3,
        # inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',
        inception_pkl=None,
        bgr2rgb=False),

    sample_kwargs=dict(sample_model='ema'))
