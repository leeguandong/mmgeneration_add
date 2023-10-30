'''
@Time    : 2022/7/14 14:43
@Author  : leeguandon@gmail.com
'''
import torch.nn as nn
import torch.nn.functional as F
from mmgen.models.losses.utils import weighted_loss
from mmgen.models.builder import MODULES


@weighted_loss
def cross_entropy(pred, target):
    return F.cross_entropy(pred, target, reduction='none')


@MODULES.register_module()
class CrossEntropy(nn.Module):
    def __init__(self, loss_weight=1.0, data_info=None, loss_name='cross_entropy'):
        super(CrossEntropy, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
                }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(weight=self.loss_weight))
            return cross_entropy(**kwargs)
        else:
            return cross_entropy(*args, weight=self.loss_weight, **kwargs)

    def loss_name(self):
        return self._loss_name
