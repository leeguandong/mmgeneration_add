'''
@Time    : 2022/6/28 14:23
@Author  : leeguandon@gmail.com
'''
from .disc_auxiliary_loss import DivergenceLoss
from .pixelwise_loss import CrossEntropy

__all__ = [
    "DivergenceLoss",
    'CrossEntropy'
]
