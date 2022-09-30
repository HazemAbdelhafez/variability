from typing import List

import torch.nn.functional
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = 'max_pool2d'


class Names:
    input_t_shape = "input_tensor_shape"
    kernel_size = 'kernel_size'
    stride = 'stride'
    padding = 'padding'
    dilation = 'dilation'
    ceil_mode = 'ceil_mode'


class Parameters(BaseParameters):

    def __init__(self, input_t_shape, kernel_size, stride, padding, dilation, ceil_mode):
        super().__init__(kernel_name)
        self.input_t_shape = input_t_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.kernel_size] = self.kernel_size
        output[Names.stride] = self.stride
        output[Names.padding] = self.padding
        output[Names.dilation] = self.dilation
        output[Names.ceil_mode] = self.ceil_mode
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters):
        kernel_parameters = cls(parameters[Names.input_t_shape], parameters[Names.kernel_size],
                                parameters[Names.stride], parameters[Names.padding],
                                parameters[Names.dilation], bool(parameters[Names.ceil_mode]))

        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        input_t_shape = kernel_params[0]
        kernel_size = kernel_params[1]
        stride = kernel_params[2]
        padding = kernel_params[3]
        dilation = kernel_params[4]
        ceil_mode = bool(kernel_params[5])
        assert len(kernel_params) == 6
        return cls(input_t_shape, kernel_size, stride, padding, dilation, ceil_mode)


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], kernel_size: List[int], stride: List[int],
                 padding: List[int], dilation: List[int], ceil_mode: bool,
                 device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)
        self.kernel_size: List[int] = kernel_size
        self.stride: List[int] = stride
        self.padding: List[int] = padding
        self.dilation: List[int] = dilation
        self.ceil_mode: bool = ceil_mode


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.max_pool2d

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.kernel_size, in_obj.stride, in_obj.padding, in_obj.dilation,
                           in_obj.ceil_mode)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.kernel_size, params.stride,
                     params.padding, params.dilation, params.ceil_mode, device)

    def generate_random_input_parameters(self):
        n = 1
        c = [8] + list(range(16, 512, 8)) + list(range(512, 1024, 128)) + [832]
        h = [3, 7] + list(range(14, 256, 7)) + [111, 71]
        c.sort()
        h.sort()

        w = h
        kernel_size = [2, 3]
        stride = [1, 2]
        padding = [0, 1]
        ceil_mode = [True, False]

        # TODO reimplement
        kernel_parameters = \
            Parameters(n, pick(c), pick(h), pick(w), pick(kernel_size), pick(stride))
        return kernel_parameters


if __name__ == '__main__':
    pass
