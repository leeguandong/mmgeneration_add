'''
@Time    : 2022/7/13 17:27
@Author  : leeguandon@gmail.com
'''
from mmgen.core.evaluation.metrics import Metric
from mmgen.core.registry import METRICS


@METRICS.register_module()
class MaxIoU(Metric):
    name = 'MaxIoU'

    def __init__(self):
        pass
