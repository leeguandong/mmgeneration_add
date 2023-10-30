'''
@Time    : 2022/7/18 14:11
@Author  : leeguandon@gmail.com
'''
import sys
import mmcv
import torch
import torchvision.transforms as T
from torch_geometric.utils import to_dense_batch
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmgen.core.evaluation.eval_hooks import GenerativeEvalHook
from mmgen_add.models.architectures.layoutgan import convert_layout_to_image


@HOOKS.register_module()
class LayoutEvalHook(GenerativeEvalHook):
    def after_train_iter(self, runner):
        interval = self.get_current_interval(runner)
        if not self.every_n_iters(runner, interval):
            return

        runner.model.eval()

        batch_size = self.dataloader.batch_size
        rank, ws = get_dist_info()
        total_batch_size = batch_size * ws

        # sample real images
        max_real_num_images = max(metric.num_images - metric.num_real_feeded
                                  for metric in self.metrics)
        # define mmcv progress bar
        if rank == 0 and max_real_num_images > 0:
            mmcv.print_log(
                f'Sample {max_real_num_images} real images for evaluation',
                'mmgen')
            pbar = mmcv.ProgressBar(max_real_num_images)

        if max_real_num_images > 0:
            for data in self.dataloader:
                label, mask = to_dense_batch(data.y, data.batch)
                bbox_real, _ = to_dense_batch(data.x, data.batch)
                padding_mask = ~mask
                self.batch_size = label.shape[0]
                self.sample_kwargs.update({'label': label})
                self.sample_kwargs.update({'padding_mask': padding_mask})
                self.sample_kwargs.update({'mask': mask})
                # self.sample_kwargs.update({'num_batches': batch_size})

                # 实际进行评估，全部换成image进行评估，原始论文是encoder后进行评估
                bbox_real = self.to_image(bbox_real, label, mask) # 3,3,60,40

                num_feed = 0
                for metric in self.metrics:
                    num_feed_ = metric.feed(bbox_real, 'reals') # 3,3,60,40
                    num_feed = max(num_feed_, num_feed)

                if num_feed <= 0:
                    break

                if rank == 0:
                    pbar.update(num_feed)

        max_num_images = max(metric.num_images for metric in self.metrics)
        if rank == 0:
            mmcv.print_log(
                f'Sample {max_num_images} fake images for evaluation', 'mmgen')

        # define mmcv progress bar
        if rank == 0:
            pbar = mmcv.ProgressBar(max_num_images)

        for _ in range(0, max_num_images, total_batch_size):

            with torch.no_grad():
                fakes = runner.model(
                    None,
                    num_batches=self.batch_size,
                    return_loss=False,
                    **self.sample_kwargs
                )

                fakes = self.to_image(fakes, self.sample_kwargs['label'], self.sample_kwargs['mask'])
                for metric in self.metrics:
                    # feed in fake images
                    metric.feed(fakes, 'fakes')

            if rank == 0:
                pbar.update(total_batch_size)

        runner.log_buffer.clear()
        # a dirty walkround to change the line at the end of pbar
        if rank == 0:
            sys.stdout.write('\n')
            for metric in self.metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    runner.log_buffer.output[name] = val

                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)

            runner.log_buffer.ready = True
        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear()

    def to_image(self, batch_bbox, batch_label, batch_mask, canvas_size=(60, 40)):
        dataset_colors = [(246, 112, 136), (173, 156, 49), (51, 176, 122), (56, 168, 197), (204, 121, 244)]
        imgs = []
        batch = batch_bbox.size(0)
        to_tensor = T.ToTensor()
        for i in range(batch):
            mask_i = batch_mask[i]
            boxes = batch_bbox[i][mask_i]
            labels = batch_label[i][mask_i]
            img = convert_layout_to_image(boxes, labels, dataset_colors, canvas_size)
            imgs.append(to_tensor(img))
        image = torch.stack(imgs)
        return image
