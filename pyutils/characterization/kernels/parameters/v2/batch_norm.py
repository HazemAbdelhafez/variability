from typing import List, Optional

import torch
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

kernel_name = 'batch_norm'


class Names:
    input_t_shape = "input_tensor_shape"
    weight_t_shape = "weight_tensor_shape"
    bias_shape = "bias_shape"
    running_mean_shape = 'running_mean_shape'
    running_var_shape = 'running_var_shape'
    training = 'training'
    momentum = 'momentum'
    eps = "eps"
    cudnn_enabled = "cudnn_enabled"


class Parameters(BaseParameters):

    def __init__(self):
        super().__init__(kernel_name)

        # Input parameters
        self.input_t_shape: List[int] = []
        self.weight_t_shape: List[int] = []

        # Add offset to the result per cell
        self.bias_shape: List[int] = []
        self.running_mean_shape: List[int] = []
        self.running_var_shape: List[int] = []

        self.training: bool = False
        self.momentum: float = 1e-5
        self.eps: float = 0.1
        self.cudnn_enabled: bool = False

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.weight_t_shape] = self.weight_t_shape
        output[Names.bias_shape] = self.bias_shape

        output[Names.running_mean_shape] = self.running_mean_shape
        output[Names.running_var_shape] = self.running_var_shape
        output[Names.training] = self.training
        output[Names.momentum] = self.momentum
        output[Names.eps] = self.eps
        output[Names.cudnn_enabled] = self.cudnn_enabled

        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters: dict):
        kernel_parameters = cls()
        kernel_parameters.input_t_shape = parameters[Names.input_t_shape]
        kernel_parameters.weight_t_shape = parameters[Names.weight_t_shape]
        kernel_parameters.bias_shape = parameters[Names.bias_shape]

        kernel_parameters.running_mean_shape = parameters[Names.running_mean_shape]
        kernel_parameters.running_var_shape = parameters[Names.running_var_shape]
        kernel_parameters.training = bool(parameters[Names.training])
        kernel_parameters.momentum = parameters[Names.momentum]
        kernel_parameters.eps = parameters[Names.eps]
        kernel_parameters.cudnn_enabled = bool(parameters[Names.cudnn_enabled])

        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        params = cls()
        params.input_t_shape = kernel_params[0]
        params.weight_t_shape = kernel_params[1]
        params.bias_shape = kernel_params[2]
        params.running_mean_shape = kernel_params[3]
        params.running_var_shape = kernel_params[4]
        params.training = bool(kernel_params[5])
        params.momentum = kernel_params[6]
        params.eps = kernel_params[7]
        params.cudnn_enabled = bool(kernel_params[8])

        return params


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], weight_t_shape: List[int], bias_shape: List[int],
                 running_mean_shape: List[int], running_var_shape: List[int], training: bool, momentum: float,
                 eps: float, cudnn_enabled: bool, device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)
        self.weight_t: Tensor = torch.randn(size=weight_t_shape, device=device)
        self.bias: Optional[Tensor] = torch.randn(size=bias_shape, device=device) if bias_shape is not None else None

        self.running_mean: Optional[Tensor] = torch.randn(size=running_mean_shape, device=device)
        self.running_var: Optional[Tensor] = torch.randn(size=running_var_shape, device=device)
        self.training: bool = bool(training)
        self.momentum: float = momentum
        self.eps: float = eps
        self.cudnn_enabled = bool(cudnn_enabled)


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.batch_norm

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.weight_t, in_obj.bias, in_obj.running_mean,
                           in_obj.running_var, in_obj.training, in_obj.momentum, in_obj.eps,
                           in_obj.cudnn_enabled)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.generator_version = 1
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, params.weight_t_shape, params.bias_shape,
                     params.running_mean_shape, params.running_var_shape,
                     params.training, params.momentum, params.eps, params.cudnn_enabled, device)

    def generate_random_input_parameters(self):
        # TODO reimplement this
        n = 1
        c = [8] + list(range(16, 1024, 32)) + list(range(1024, 2560, 128)) + [2560]
        h = [3, 7] + list(range(14, 256, 7)) + [149]
        c.sort()
        h.sort()
        w = h
        c_pick = pick(c)
        kernel_parameters = Parameters()
        return kernel_parameters


if __name__ == '__main__':
    pass
