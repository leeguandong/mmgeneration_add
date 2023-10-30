'''
@Time    : 2022/7/13 14:53
@Author  : leeguandon@gmail.com
'''
import torch
from mmgen.datasets.builder import PIPELINES


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


@PIPELINES.register_module()
class LexicographicSort:
    def __call__(self, data):
        assert not data.attr['has_canvas_element']
        l, t, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data


@PIPELINES.register_module()
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if self.p < torch.rand(1):
            return data

        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data
