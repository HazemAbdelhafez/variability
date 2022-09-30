from typing import List

import torch.nn.functional
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = "adaptive_avg_pool2d"


class Names:
    input_t_shape = "input_tensor_shape"
    output_t_size = "output_tensor_size"


class Parameters(BaseParameters):

    def __init__(self, input_t_shape, output_t_size):
        super().__init__(kernel_name)
        self.input_t_shape = input_t_shape
        self.output_t_size = output_t_size

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.output_t_size] = self.output_t_size
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters):
        kernel_parameters = cls(parameters[Names.input_t_shape], parameters[Names.output_t_size])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        return cls(kernel_params[0], kernel_params[1])


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], output_t_size: List[int], device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)
        self.output_t_size = output_t_size


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.nn.functional.adaptive_avg_pool2d

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.output_t_size)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.generator_version = 1
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.output_t_size, device)

    def generate_random_input_parameters(self):
        # TODO: reimplement this.
        n = 1
        c = [64] + list(range(128, 2624, 64))
        h = list(range(3, 19, 1))
        w = h
        out_h = list(range(1, 10, 1))
        out_w = out_h

        pick_h = pick(h)
        pick_w = pick(w)
        pick_out_h = pick(out_h)
        pick_out_w = pick(out_w)

        kernel_parameters = Parameters(n, c)
        return kernel_parameters


if __name__ == '__main__':
    pass
