'''
@Time    : 2022/7/12 16:43
@Author  : leeguandon@gmail.com
'''
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors
from torch_geometric.utils import to_dense_batch
import torchvision.transforms as T
from mmgen.models.gans.base_gan import BaseGAN
from mmgen.models.builder import MODELS, build_module
from mmgen_add.models.architectures.layoutgan import convert_layout_to_image


@MODELS.register_module()
class LayoutGAN(BaseGAN):
    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super(LayoutGAN, self).__init__()
        self._gen_cfg = deepcopy(generator)
        self.generator = build_module(generator)

        if discriminator is not None:
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        # label: [4,15],mask: [4,15],bbox_real:[4,15,4]
        # tensor([[1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        # [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        label, mask = to_dense_batch(data_batch.y, data_batch.batch)  # label:4,17 / mask:4,17
        bbox_real, _ = to_dense_batch(data_batch.x, data_batch.batch)  # bbox_real: 4,17,4
        padding_mask = ~mask  # 4,17
        batch_size = label.shape[0]  # 4

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # update G network
        optimizer['generator'].zero_grad()
        disc_pred_bbox_fake_g = self.generator(None, label, padding_mask)  # 4,15,4
        disc_pred_bbox_fake = self.discriminator(disc_pred_bbox_fake_g, label, padding_mask)  # 4
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=disc_pred_bbox_fake,
            disc_pred_fake_g=disc_pred_bbox_fake_g,
            label=label,
            iteration=curr_iter,
            loss_scaler=loss_scaler)

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_gen, optimizer['generator'],
                    loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['generator'].step()

        # update D network
        optimizer['discriminator'].zero_grad()
        disc_pred_fake = self.discriminator(disc_pred_bbox_fake_g.detach(), label, padding_mask)
        disc_real, logit_cls, bbox_recon = self.discriminator(bbox_real, label, padding_mask, reconst=True)
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_real,
            logit_cls=logit_cls,
            target=data_batch.y,
            bbox_recon=bbox_recon,
            bbox_real=data_batch.x,
            iteration=curr_iter,
            loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp

            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['discriminator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        fake_imgs = self.save_image(disc_pred_bbox_fake_g, label, mask) # disc_pred_fake_g:4,17,4 / label:4,17 / mask:4,17 /fake_imgs:4,3,60,40
        real_imgs = self.save_image(bbox_real, label, mask) # bbox_real: 4,17,4 / real_imgs: 4,3,60,40

        results = dict(fake_imgs=fake_imgs, real_imgs=real_imgs)
        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def save_image(self, batch_bbox, batch_label, batch_mask, canvas_size=(60, 40)):
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

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          sample_model='ema/orig',
                          **kwargs):
        """Sample images from noises by using the generator.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional):  The number of batch size.
                Defaults to 0.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with queried
                data, including generated images, will be returned.
        """
        if sample_model == 'ema':
            assert self.use_ema
            _model = self.generator_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator_ema
        else:
            _model = self.generator

        outputs = _model(noise, **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator
            outputs_ = _model(noise, **kwargs)

            if isinstance(outputs_, dict):
                outputs['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs
