'''
@Time    : 2022/7/12 16:00
@Author  : leeguandon@gmail.com
'''
dataset_type = 'Magazine'

train_pipeline = [
    dict(type='LexicographicSort'),  # 这块和原始的mm并没有完全对齐
    dict(type='RandomHorizontalFlip', p=0.5)
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(type=dataset_type,
               img_root=r"E:\comprehensive_library\mmgeneration_add\data\MagImage",
               split='train', pipeline=train_pipeline, test_mode=False),
    val=dict(type=dataset_type,
             img_root=r"E:\comprehensive_library\mmgeneration_add\data\MagImage",
             split='val', pipeline=None, test_mode=True)
)
