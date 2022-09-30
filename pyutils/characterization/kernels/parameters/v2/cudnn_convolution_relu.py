from random import choices
from typing import Optional, List

import torch
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.characterization.kernels.utils.resources import get_unique_parameters
from pyutils.characterization.networks.analyzers.runtime import RuntimeAnalyzer
from pyutils.common.config import Timers
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = "cudnn_convolution_relu"


class Names:
    input_t_shape = "input_tensor_shape"
    weight_t_shape = "weight_tensor_shape"
    bias_shape = "bias_shape"
    stride = "stride"
    padding = "padding"
    dilation = "dilation"
    groups = "groups"


class Parameters(BaseParameters):

    def __init__(self, child_kernel_name=None):
        if child_kernel_name is None:
            super().__init__(kernel_name)
        else:
            super().__init__(child_kernel_name)

        # Input parameters
        self.input_t_shape: List[int] = []
        self.weight_t_shape: List[int] = []

        # Add offset to the conv result per cell
        self.bias_shape: List[int] = []

        # Step size for moving the filter across the input cells
        self.stride: List[int] = [1, 1]

        # How many rows/columns to add to the 2D input for padding
        self.padding: List[int] = [0, 0]

        # How far are the filter cells from each other when being applied to the input
        self.dilation: List[int] = [1, 1]
        self.groups: int = 1

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.weight_t_shape] = self.weight_t_shape
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

        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):

        params = cls()
        params.input_t_shape = kernel_params[0]
        params.weight_t_shape = kernel_params[1]
        params.bias_shape = kernel_params[2]

        params.stride = kernel_params[3]
        params.padding = kernel_params[4]
        params.dilation = kernel_params[5]
        params.groups = kernel_params[6]
        assert (params.weight_t_shape[1] == int(params.input_t_shape[1] / params.groups))

        return params


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], weight_t_shape: List[int], bias_shape: Optional[List[int]],
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


class Module(BaseModule):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.kernel = torch.cudnn_convolution_relu

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.weight_t, in_obj.bias,
                           in_obj.stride, in_obj.padding, in_obj.dilation, in_obj.groups)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.weight_t_shape, params.bias_shape, params.stride, params.padding,
                     params.dilation, params.groups, device)

    def generate_random_input_parameters(self):
        return self._generate_random_input_parameters_v5()

    @staticmethod
    def _generate_random_input_parameters_v5():
        # Pick real-life configs with higher probability than random ones
        versions = [1, 2]
        version = choices(versions, weights=[0.5, 0.5], k=1)[0]
        if version == 1:
            return Generator._generate_random_input_parameters_v1()
        else:
            return Generator._generate_random_input_parameters_v2()

    @staticmethod
    def _generate_random_input_parameters_v4():
        version = 4
        n = [1]
        c = [3] + list(range(16, 2160, 256))
        h = [7] + list(range(14, 301, 21))

        out_channels = [3] + list(range(8, 1281, 16))
        kernel_dim = list(range(1, 13, 2))  # 1, 3, 5, 7, 11
        stride_dim = [1, 2, 4]
        padding_dim = [0, 1, 2, 3]
        groups = [1, 24] + list(range(58, 960, 58)) + [960]
        bias = [0, 1]

        c.sort()
        h.sort()
        out_channels.sort()
        groups.sort()

        # from observed ranges
        max_weight_size = 8294400

        while True:
            w_choice = h_choice = pick(h)
            in_channels_choice = c_choice = pick(c)
            parameters = Parameters()
            parameters.n = pick(n)
            parameters.c = c_choice
            parameters.h = h_choice
            parameters.w = w_choice
            parameters.in_channels = in_channels_choice

            if in_channels_choice in groups:
                parameters.groups = in_channels_choice
                parameters.out_channels = in_channels_choice
            else:
                parameters.groups = 1
                parameters.out_channels = pick(out_channels)

            kernel_dim_choice = min(pick(kernel_dim), parameters.h)
            parameters.kernel = (kernel_dim_choice, kernel_dim_choice)

            stride_dim_choice = pick(stride_dim)
            parameters.stride = (stride_dim_choice, stride_dim_choice)

            padding_dim_choice = pick(padding_dim)
            parameters.padding = (padding_dim_choice, padding_dim_choice)

            parameters.dilation = (1, 1)
            parameters.bias = pick(bias)

            if kernel_dim_choice * kernel_dim_choice * parameters.in_channels * \
                    parameters.out_channels < max_weight_size:
                break

        # Mark the generator version used to create these parameters
        parameters.generator_version = version
        return parameters

    @staticmethod
    def _generate_random_input_parameters_v1():
        version = 1
        values = get_unique_parameters(kernel_name)
        parameters = Parameters.from_dict(pick(values))

        # Mark the generator version used to create these parameters
        parameters.generator_version = version
        return parameters

    @staticmethod
    def _generate_random_input_parameters_v2():
        version = 2

        values = list()
        parameters = Parameters().from_dict(pick(values))

        # Mark the generator version used to create these parameters
        parameters.generator_version = version
        return parameters


if __name__ == "__main__":
    g = Generator()
    m = g.create_module()
    """
    cudnn_convolution_relu 1_2064_7_7-192_2064_1_1-192-1_1-0_0-1_1-1
    cudnn_convolution_relu 1_2112_7_7-192_2112_1_1-192-1_1-0_0-1_1-1
    cudnn_convolution_relu 1_2160_7_7-192_2160_1_1-192-1_1-0_0-1_1-1
    """
    custom_params = Parameters.from_list([[1, 2160, 7, 7], [192, 2160, 1, 1], None, [1, 1], [0, 0], [1, 1], 1])
    inp = g.create_input(custom_params)
    runtime_ms_mean, _ = RuntimeAnalyzer.time_a_kernel(kernel_name, custom_params, Timers.CUEvents.cuda_graphs, 10)
