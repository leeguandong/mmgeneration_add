import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def divergence_loss(discriminator,
                    real_data,
                    fake_data,
                    p=6,
                    k=2,
                    mask=None,
                    norm_mode='pixel'):
    """Calculate gradient penalty for wgan-div.

    In the detailed implementation, there are two streams where one uses the
    pixel-wise gradient norm, but the other adopts normalization along instance
    (HWC) dimensions. Thus, ``norm_mode`` are offered to define which mode you
    want.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        mask (Tensor): Masks for inpainting. Default: None.
        norm_mode (str): This argument decides along which dimension the norm
            of the gradients will be calculated. Currently, we support ["pixel"
            , "HWC"]. Defaults to "pixel".

    Returns:
        Tensor: A tensor for gradient penalty.
    """
    batch_size = real_data.size(0)

    real_data = autograd.Variable(real_data, requires_grad=True)
    fake_data = autograd.Variable(fake_data, requires_grad=True)
    real_disc = discriminator(real_data)
    fake_disc = discriminator(fake_data)

    # Compute W-div gradient penalty
    # real_grad_out = torch.full((real_data.size(0),), 1, dtype=torch.float32, requires_grad=False)
    # fake_grad_out = torch.full((fake_data.size(0),), 1, dtype=torch.float32, requires_grad=False)
    # real_grad_out = Variable(Tensor(real_data.size(0), 1).fill_(1.0), requires_grad=False)
    # fake_grad_out = Variable(Tensor(fake_data.size(0), 1).fill_(1.0), requires_grad=False)

    real_gradient = autograd.grad(
        outputs=real_disc,
        inputs=real_data,
        # grad_outputs=real_grad_out,
        grad_outputs=torch.ones_like(real_disc),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    fake_gradient = autograd.grad(
        outputs=fake_disc,
        inputs=fake_data,
        # grad_outputs=fake_grad_out,
        grad_outputs=torch.ones_like(fake_disc),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    real_gradient_norm = real_gradient.view(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_gradient_norm = fake_gradient.view(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

    if mask is not None:
        real_gradient_norm = real_gradient_norm * mask
        fake_gradient_norm = fake_gradient_norm * mask

    if norm_mode == 'pixel':
        # gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradients_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    elif norm_mode == 'HWC':
        # gradients_penalty = ((gradients.reshape(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()
        gradients_penalty = torch.mean(real_gradient_norm.reshape(batch_size, -1) +
                                       fake_gradient_norm.reshape(batch_size, -1)) * k / 2
    else:
        raise NotImplementedError(
            'Currently, we only support ["pixel", "HWC"] '
            f'norm mode but got {norm_mode}.')
    if mask is not None:
        gradients_penalty /= torch.mean(mask)

    return gradients_penalty


@MODULES.register_module()
class DivergenceLoss(nn.Module):
    """Gradient Penalty for WGAN-DIV.

    In the detailed implementation, there are two streams where one uses the
    pixel-wise gradient norm, but the other adopts normalization along instance
    (HWC) dimensions. Thus, ``norm_mode`` are offered to define which mode you
    want.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we will need to provide ``discriminator``, ``real_data``,
    and ``fake_data`` as input. Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            discriminator='disc',
            real_data='real_imgs',
            fake_data='fake_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        norm_mode (str): This argument decides along which dimension the norm
            of the gradients will be calculated. Currently, we support ["pixel"
            , "HWC"]. Defaults to "pixel".
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_gp'.
    """

    def __init__(self,
                 k=2,
                 p=6,
                 loss_weight=1.0,
                 norm_mode='pixel',
                 data_info=None,
                 loss_name='loss_div'):
        super().__init__()
        self.k = k
        self.p = p
        self.loss_weight = loss_weight
        self.norm_mode = norm_mode
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gradient_penalty_loss``.
        """
        kwargs.update(dict(k=self.k))
        kwargs.update(dict(p=self.p))
        # use data_info to build computational path
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
            kwargs.update(
                dict(weight=self.loss_weight, norm_mode=self.norm_mode))
            return divergence_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return divergence_loss(
                *args, weight=self.loss_weight, **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
