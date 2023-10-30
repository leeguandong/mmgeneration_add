model = dict(
    type='LayoutGAN',
    generator=dict(
        type='LayoutNetppGenerator',
        dim_latent=4,
        num_label=5,
        d_model=256,
        nhead=4,
        num_layers=8),
    discriminator=dict(
        type='LayoutNetppDiscriminator',
        num_label=5,
        d_model=256,
        nhead=4,
        num_layers=8,
        max_bbox=50),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=[
        dict(
            type='MSELoss',
            loss_weight=10,
            data_info=dict(pred='bbox_recon', target='bbox_real')),
        dict(
            type='CrossEntropy',
            loss_weight=1,
            data_info=dict(pred='logit_cls', target='target'))
    ])
train_cfg = dict(use_ema=True)
test_cfg = None
optimizer = dict(
    generator=dict(type='Adam', lr=1e-05),
    discriminator=dict(type='Adam', lr=1e-05))
dataset_type = 'Magazine'
train_pipeline = [
    dict(type='LexicographicSort'),
    dict(type='RandomHorizontalFlip', p=0.5)
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        type='Magazine',
        img_root='E:\comprehensive_library\mmgeneration_add\data\MagImage',
        split='train',
        pipeline=[
            dict(type='LexicographicSort'),
            dict(type='RandomHorizontalFlip', p=0.5)
        ],
        test_mode=False),
    val=dict(
        type='Magazine',
        img_root='E:\comprehensive_library\mmgeneration_add\data\MagImage',
        split='val',
        pipeline=None,
        test_mode=True))
checkpoint_config = dict(interval=500, by_epoch=False, max_keep_ckpts=30)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='visual',
        interval=1,
        rerange=False,
        bgr2rgb=False,
        res_name_list=['fake_imgs', 'real_imgs'])
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
total_iters = 30000
evaluation = dict(
    type='LayoutEvalHook',
    interval=1,
    metrics=dict(type='FID', num_images=3, inception_pkl=None, bgr2rgb=False),
    sample_kwargs=dict(sample_model='ema'))
work_dir = './results'
gpu_ids = [0]
