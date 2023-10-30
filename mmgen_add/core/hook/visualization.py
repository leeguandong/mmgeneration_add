# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image


@HOOKS.register_module()
class VisualizationLayoutHook(Hook):
    def __init__(self,
                 output_dir,
                 res_name_list,
                 interval=-1,
                 filename_tmpl='iter_{}_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=1,
                 padding=4):
        assert mmcv.is_list_of(res_name_list, str)
        self.output_dir = output_dir
        self.res_name_list = res_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

    @master_only
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        results = runner.outputs['results']

        if not hasattr(self, '_out_dir'):
            self._out_dir = osp.join(runner.work_dir, self.output_dir)
        mmcv.mkdir_or_exist(self._out_dir)

        # results[]
        img_list = [results[k] for k in self.res_name_list if k in results]
        for index, img in enumerate(img_list):
            if self.nrow is None:
                self.nrow = int(np.ceil(np.sqrt(img.shape[0])))
            filename = self.filename_tmpl.format(runner.iter + 1, index)

            save_image(
                img,
                osp.join(self._out_dir, filename),
                nrow=self.nrow,
                padding=self.padding)
