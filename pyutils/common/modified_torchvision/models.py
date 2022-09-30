import torch
from torch import Tensor
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.googlenet import model_urls as googlenet_model_urls, GoogLeNet
from torchvision.models.inception import model_urls as inception_model_urls, Inception3


# GoogleNet and Inception return namedtuple. This causes an exception in scripted models when we call result.cpu()
# a workaround that avoids if conditions is to override the forward method and use inherited classes.

class TensorBasedInceptionV3(Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=None):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits,
                         transform_input=transform_input, inception_blocks=inception_blocks,
                         init_weights=init_weights)

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        if torch.jit.is_scripting():
            return x
        else:
            return self.eager_outputs(x, aux)


def tensor_based_inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        kwargs['init_weights'] = False  # we are loading weights from a pretrained model
        model = TensorBasedInceptionV3(**kwargs)
        state_dict = load_state_dict_from_url(inception_model_urls['inception_v3_google'], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return TensorBasedInceptionV3(**kwargs)


class TensorBasedGoogleNet(GoogLeNet):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=None,
                 blocks=None):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input,
                         init_weights=init_weights, blocks=blocks)

    def forward(self, x) -> Tensor:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        return x
        # Temp solution to allow jit.trace to work with GoogleNet.
        # if torch.jit.is_scripting():
        #     return x
        # else:
        #     raise Exception("TensorBasedGoogleNet must be used in scripting mode only.")


def tensor_based_googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        # if kwargs['aux_logits']:
        #     warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
        #                   'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = TensorBasedGoogleNet(**kwargs)
        state_dict = load_state_dict_from_url(googlenet_model_urls['googlenet'], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        return model

    return TensorBasedGoogleNet(**kwargs)


class MatMulModule(nn.Module):
    def __init__(self):
        super(MatMulModule, self).__init__()
        self.kernel = torch.matmul

    def forward(self, x):
        return self.kernel(x[0], x[1])
