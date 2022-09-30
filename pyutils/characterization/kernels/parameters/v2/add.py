import ast
from typing import List

import torch

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

MAX_NUM_DIMS = 4

kernel_name = "add"


class Names:
    input_t_shape = 'input_shape'
    other_t_shape = 'other_shape'
    alpha = 'alpha'


class Parameters(BaseParameters):

    def __init__(self, input_t_shape: List, other_t_shape: List, alpha=1):
        super().__init__(kernel_name)

        # We assume 4D tensors at max.
        assert len(input_t_shape) <= MAX_NUM_DIMS
        assert len(other_t_shape) <= MAX_NUM_DIMS

        self.input_t_shape = [0 for _ in range(MAX_NUM_DIMS - len(input_t_shape))] + input_t_shape
        self.other_t_shape = [0 for _ in range(MAX_NUM_DIMS - len(other_t_shape))] + other_t_shape

        # Element wise addition so input and other must have the same dimensions
        self.alpha = alpha

    def to_str(self):
        str_form = "-".join([str(i) for i in self.input_t_shape + self.other_t_shape])
        return str_form

    def to_dict(self):
        output = dict()
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.other_t_shape] = self.other_t_shape
        output[Names.alpha] = self.alpha
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    def to_id(self):
        return self.to_list(exclude=[Names.alpha])

    @classmethod
    def from_str(cls, params: str):
        """ Input is a single line.
        e.g.,
            [1000]-[1, 1000]-1
            [1, 24, 56, 56]-[1, 24, 56, 56]-1
        """
        chunks = [ast.literal_eval(i) for i in params.split('-')]
        for j in range(len(chunks)):
            chunk = chunks[j]
            if isinstance(chunk, list):
                chunks[j] = [i for i in chunk if i != 0]

        return cls(chunks[0], chunks[1], chunks[2])

    @classmethod
    def from_dict(cls, parameters: dict) -> BaseParameters:
        kernel_parameters = cls(parameters[Names.input_t_shape], parameters[Names.other_t_shape],
                                parameters[Names.alpha])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list) -> BaseParameters:
        assert isinstance(kernel_params[0], list)
        assert isinstance(kernel_params[1], list)
        assert isinstance(kernel_params[2], (int, float))
        return cls(kernel_params[0], kernel_params[1], kernel_params[2])


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.add

    def forward(self, kernel_input: List[torch.Tensor]):
        return self.kernel(kernel_input[0], kernel_input[1])


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.generator_version = 1
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return [torch.randn(params.input_t_shape, device=device),
                torch.randn(params.other_t_shape, device=device)]

    def generate_random_input_parameters(self):
        # TODO: reimplement this.
        n = 1
        c = [6, 12, 24] + list(range(32, 192, 16))
        for i in [1000, 4096, 24, 32, 64, 96, 160, 40, 80, 192]:
            if i not in c:
                c.append(i)
        c.append(c[-1] + 16)
        c.append(c[-1] + 16)

        h = [0] + list(range(7, 56 + 14, 7))
        for i in [0, 56, 28, 14, 14, 7, 28, 14, 7]:
            if i not in h:
                h.append(i)
        c.sort()
        h.sort()
        picked_c = pick(c)
        picked_w = picked_h = pick(h)

        kernel_parameters = Parameters(n, picked_c, picked_h, picked_w)
        return kernel_parameters


if __name__ == '__main__':
    pass
