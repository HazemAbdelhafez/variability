from typing import List

import torch.nn.functional
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = "relu"


class Names:
    input_t_shape = "input_t_shape"
    inplace = 'inplace'


class Parameters(BaseParameters):

    def __init__(self, input_t_shape, inplace=True):
        super().__init__(kernel_name)
        self.input_t_shape = input_t_shape
        self.inplace = inplace

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.inplace] = self.inplace
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters):
        kernel_parameters = cls(parameters[Names.input_t_shape], parameters[Names.inplace])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        input_shape = kernel_params[0]
        if len(kernel_params) == 2:
            inplace = kernel_params[1]
        else:
            inplace = True
        return cls(input_shape, inplace)


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], inplace: bool, device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)
        self.inplace: bool = inplace


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.nn.functional.relu

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.inplace)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.inplace, device)

    def generate_random_input_parameters(self):
        # TODO: re-implement this.
        n = 1
        c = [8] + list(range(16, 592, 8)) + list(range(592, 1536, 8)) + list(range(1536, 4112, 8))
        h = [0] + list(range(7, 231, 7)) + [13]
        c.sort()
        h.sort()
        w = h
        picked_h = pick(h)
        if picked_h == 0:
            picked_w = 0
        else:
            picked_w = pick(w)

        kernel_parameters = Parameters(n, pick(c))
        return kernel_parameters


if __name__ == '__main__':
    pass
