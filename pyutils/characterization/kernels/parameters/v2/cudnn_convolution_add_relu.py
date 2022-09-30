from typing import List, Optional

import torch
from torch import Tensor

import pyutils.characterization.kernels.parameters.v2.cudnn_convolution_relu as cu_conv_relu
from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = "cudnn_convolution_add_relu"


class Names(cu_conv_relu.Names):
    added_t_shape = "added_tensor_shape"
    alpha = "alpha"


class Parameters(cu_conv_relu.Parameters):

    def __init__(self):
        super().__init__(kernel_name)

        self.added_t_shape: List[int] = []
        self.alpha: int = 1

    def to_dict(self):
        # We override this method to specify the order of keys in the dict.
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.weight_t_shape] = self.weight_t_shape
        output[Names.added_t_shape] = self.added_t_shape
        output[Names.alpha] = self.alpha
        output[Names.bias_shape] = self.bias_shape
        output[Names.stride] = self.stride
        output[Names.padding] = self.padding
        output[Names.dilation] = self.dilation
        output[Names.groups] = self.groups
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters: dict):
        """
        Parse kernel parameters from a dictionary and creates and returns Parameters object from it
        """
        kernel_parameters = cls()
        kernel_parameters.input_t_shape = parameters[Names.input_t_shape]
        kernel_parameters.weight_t_shape = parameters[Names.weight_t_shape]
        kernel_parameters.bias_shape = parameters[Names.bias_shape]
        kernel_parameters.stride = parameters[Names.stride]
        kernel_parameters.padding = parameters[Names.padding]
        kernel_parameters.dilation = parameters[Names.dilation]
        kernel_parameters.groups = parameters[Names.groups]
        kernel_parameters.added_t_shape = parameters[Names.added_t_shape]
        kernel_parameters.alpha = parameters[Names.alpha]
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        params = cls()
        params.input_t_shape = kernel_params[0]
        params.weight_t_shape = kernel_params[1]
        params.added_t_shape = kernel_params[2]
        params.alpha = kernel_params[3]
        params.bias_shape = kernel_params[4]

        params.stride = kernel_params[5]
        params.padding = kernel_params[6]
        params.dilation = kernel_params[7]
        params.groups = kernel_params[8]
        assert (params.weight_t_shape[1] == int(params.input_t_shape[1] / params.groups))

        return params


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], weight_t_shape: List[int], added_t_shape: List[int], alpha: int,
                 bias_shape: List[int],
                 stride: List[int], padding: List[int], dilation: List[int], groups: int,
                 device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)
        self.weight_t: Tensor = torch.randn(size=weight_t_shape, device=device)

        self.bias: Optional[Tensor] = torch.randn(size=bias_shape, device=device) if bias_shape is not None else None

        # Step size for moving the filter across the input cells
        self.stride: List[int] = stride

        # How many rows/colums to add to the 2D input for padding
        self.padding: List[int] = padding

        # How far are the filter cells from each other when being applied to the input
        self.dilation: List[int] = dilation
        self.groups: int = groups

        self.added_t = torch.randn(size=added_t_shape, device=device)
        self.alpha = alpha


class Module(BaseModule):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.kernel = torch.cudnn_convolution_add_relu

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.weight_t, in_obj.added_t, in_obj.alpha, in_obj.bias,
                           in_obj.stride, in_obj.padding, in_obj.dilation, in_obj.groups)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.weight_t_shape, params.added_t_shape, params.alpha,
                     params.bias_shape, params.stride, params.padding,
                     params.dilation, params.groups, device)

    def generate_random_input_parameters(self):
        pass


if __name__ == "__main__":
    m = Generator()
    m.create_module()
