# define GAN model


model = dict(
    type='LayoutGAN',
    generator=dict(
        type='LayoutNetppGenerator',
        dim_latent=4,  # Need to be set.
        num_label=5,  # num_classes
        d_model=256,
        nhead=4,
        num_layers=8
    ),
    discriminator=dict(
        type='LayoutNetppDiscriminator',
        num_label=5,  # num_classes
        d_model=256,
        nhead=4,
        num_layers=8,
        max_bbox=50
    ),
    gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
    disc_auxiliary_loss=[
        dict(
            type='MSELoss',
            loss_weight=10,
            data_info=dict(pred='bbox_recon', target='bbox_real')),
        dict(
            type='CrossEntropy',
            loss_weight=1,
            data_info=dict(pred='logit_cls', target='target')
        )
    ]
)

train_cfg = dict(use_ema=True)
test_cfg = None

# define optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=1e-5),
    discriminator=dict(type='Adam', lr=1e-5))
