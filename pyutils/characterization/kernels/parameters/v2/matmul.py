from typing import List

import torch
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = "matmul"


class Names:
    mat1_shape = 'mat1_shape'
    mat2_shape = 'mat2_shape'


class Parameters(BaseParameters):
    def __init__(self, mat1_shape, mat2_shape):
        super().__init__(kernel_name)

        self.mat1_shape: List[int] = mat1_shape
        self.mat2_shape: List[int] = mat2_shape

        self.alpha = 1
        self.beta = 1

    def to_dict(self):
        output = dict()
        output[Names.mat1_shape] = self.mat1_shape
        output[Names.mat2_shape] = self.mat2_shape
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters: dict):
        kernel_parameters = cls(parameters[Names.mat1_shape], parameters[Names.mat2_shape])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        mat1_shape = kernel_params[0]
        mat2_shape = kernel_params[1]
        params = cls(mat1_shape, mat2_shape)
        return params


class Input:
    @torch.jit.ignore
    def __init__(self, mat1_shape: List[int], mat2_shape: List[int], device: torch.device = 'cuda'):
        self.mat1: Tensor = torch.randn(size=mat1_shape, device=device)
        self.mat2: Tensor = torch.randn(size=mat2_shape, device=device)


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.mm

    def forward(self, kernel_input: Input):
        return self.kernel(kernel_input.mat1, kernel_input.mat2)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.mat1_shape, params.mat2_shape, device)

    def generate_random_input_parameters(self):
        # TODO: reimplement
        pass
        # mat1_r = 1
        # mat1_c = list(range(256, 4096, 256)) + list(range(8192, 10240 + 256, 256)) + list(
        #     range(25088 - 2048, 25088 + 2048, 128)) + [1280, 2208]
        # mat2_c = [512, 1000] + list(range(1024, 4096 + 512, 256))
        #
        # mat1_c.sort()
        # mat2_c.sort()
        #
        # # Picked values
        # picked_mat1_c = pick(mat1_c)
        # picked_mat2_r = picked_mat1_c
        # picked_mat2_c = pick(mat2_c)
        # picked_bias_size = picked_mat2_c
        #
        # kernel_parameters = Parameters(mat1_r, picked_mat1_c, picked_mat2_r, picked_mat2_c, picked_bias_size)
        # return kernel_parameters


if __name__ == '__main__':
    g = Generator()
    g.create_module()
