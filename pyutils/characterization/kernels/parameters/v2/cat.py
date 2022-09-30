import ast
from typing import List, Tuple

import torch

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.common.utils import pick
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION

MAX_OUTPUT_SIZE = 1204224  # Calculated manually from the vision models data

kernel_name = "cat"


class Names:
    input_tensors_list = 'input_tensors'
    dim = 'dim'
    count = 'count'
    tensor_shape = "tensor_shape"


class MultipleTensors:
    """
    Example of cat operation from the stack file:
    aten::cat|[[1, 192, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28],[1, 48, 28, 28]]|1|

    we use multiple tensors to summarize the above arguments as follows
    aten::cat|[MultipleTensors(shape=[1, 192, 28, 28], count=1),MultipleTensors(shape=[1, 48, 28, 28], count=1)]|1|
    This is a more compact form.

    """

    def __init__(self, count: int, shape: List[int]):
        self.count = count
        self.shape = shape

    def __str__(self):
        str_form = '-'.join([str(x) for x in [self.count] + self.shape])
        return str_form

    def to_dict(self):
        output = dict()
        output[Names.count] = self.count
        output[Names.tensor_shape] = self.shape
        return output

    def to_list(self):
        output = [self.shape for _ in range(self.count)]
        return output

    @staticmethod
    def from_dict(description):
        count = description[Names.count]
        shape = description[Names.tensor_shape]
        return MultipleTensors(count, shape)


class Parameters(BaseParameters):
    def __init__(self, input_tensors_description: List[MultipleTensors] = None, dim=1):
        super().__init__(kernel_name)
        self.input_tensors_description = input_tensors_description
        self.dim = dim  # Dimension to concatenate along
        self.count = len(input_tensors_description)  # Number of input tensors descriptors

    def to_str(self):
        str_form = f'c{len(self.input_tensors_description)}' + '_'
        for item in self.input_tensors_description:
            str_form += str(item) + '_'
        str_form = str_form.strip('_')
        str_form += "_d"
        str_form += str(self.dim)
        return str_form

    def to_dict(self):
        output = dict()
        output[Names.input_tensors_list] = [i.to_dict() for i in self.input_tensors_description]
        output[Names.dim] = self.dim
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    def to_list(self, exclude: list = ()):
        output = list()
        for i in self.input_tensors_description:
            output.append(i.to_list())
        output.append(self.dim)
        return output

    @classmethod
    def from_dict(cls, parameters):
        input_tensors_description = list()
        for i in parameters[Names.input_tensors_list]:
            input_tensors_description.append(MultipleTensors.from_dict(i))
        kernel_parameters = cls(input_tensors_description, parameters[Names.dim])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):

        input_tensors = kernel_params[0]
        dim = kernel_params[1]

        # Create a string representation of the input tensors to extract the unique shapes using a dict.
        tmp = dict()
        str_repr = [str(i) for i in input_tensors]
        for i in str_repr:
            tmp[i] = tmp.get(i, 0) + 1

        input_tensors_description = list()
        for tensor_shape, count in tmp.items():
            input_tensor_description = \
                MultipleTensors(count, ast.literal_eval(tensor_shape))
            input_tensors_description.append(input_tensor_description)

        return cls(input_tensors_description, dim=dim)


class Input:
    @torch.jit.ignore
    def __init__(self, input_tensors_description: List[MultipleTensors], dim: int, device: torch.device = 'cuda'):
        self.dim = dim
        self.tensors_list = list()
        for tensor_description in input_tensors_description:
            single_tensor = torch.randn(size=tensor_description.shape, device=device)
            for count in range(tensor_description.count):
                self.tensors_list.append(single_tensor)


class Module(BaseModule):
    def __init__(self):
        super(Module, self).__init__(name=kernel_name)
        self.kernel = torch.cat

    def forward(self, x: Tuple[List[torch.Tensor], int]):
        return self.kernel(x[0], x[1])


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.module = Module()

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        tmp = Input(params.input_tensors_description, params.dim, device)
        return tmp.tensors_list, tmp.dim

    def generate_random_input_parameters(self):
        dim = 1
        unique_tensor_count = [1, 2, 3, 4]
        duplicate_tensor_count = list(range(1, 37))

        n = [1]
        c = list(range(32, 384 + 32, 32))
        for i in [1, 32, 48, 58, 64, 96, 112, 116, 128, 160, 192, 208, 224, 232, 256, 288, 320, 384, 768, 1056]:
            if i not in c:
                c.append(i)
        c.append(c[-1] + 32)  # To cover larger values than seen before
        c.append(c[-1] + 32)  # To cover larger values than seen before
        c.append(c[-1] + 32)  # To cover larger values than seen before

        h = list(range(7, 56 + 7, 7))
        for i in [7, 8, 13, 14, 17, 27, 28, 35, 55, 56, 299]:
            if i not in h:
                h.append(i)
        h.append(h[-1] + 7)  # To cover larger values than seen before
        h.append(h[-1] + 7)  # To cover larger values than seen before
        h.append(h[-1] + 7)  # To cover larger values than seen before
        h.append(3)  # To cover smaller values than seen before
        h.append(5)  # To cover smaller values than seen before

        c.sort()
        h.sort()

        while True:
            picked_unique_tensor_count = pick(unique_tensor_count)
            input_tensors = list()
            picked_n = pick(n)
            picked_w = picked_h = pick(h)
            output_channels = 0
            for i in range(picked_unique_tensor_count):
                picked_duplicate_tensor_count = pick(duplicate_tensor_count)
                picked_c = pick(c)
                output_channels += picked_c
                multiple_tensors_description = \
                    MultipleTensors(picked_duplicate_tensor_count, [picked_n, picked_c, picked_h, picked_w])
                input_tensors.append(multiple_tensors_description)
            if picked_n * output_channels * picked_h * picked_w <= MAX_OUTPUT_SIZE:
                return Parameters(input_tensors, dim)

    @staticmethod
    def analyze_data_from_models():
        parameters_list = Analysis.get_unique_parameters()
        dims = set([x.dim for x in parameters_list])
        print('dims: ', dims)

        unique_tensor_count = set([x.count for x in parameters_list])
        print('unique_count: ', unique_tensor_count)

        duplicate_tensor_count = list()
        n = list()
        c = list()
        h = list()
        w = list()
        for parameters in parameters_list:
            for tensor_description in parameters.input_tensors_description:
                duplicate_tensor_count.append(tensor_description.count)
                n.append(tensor_description.shape[0])
                c.append(tensor_description.shape[1])
                h.append(tensor_description.shape[2])
                w.append(tensor_description.shape[3])

        duplicate_tensor_count = list(set(duplicate_tensor_count))
        n = list(set(n))
        c = list(set(c))
        h = list(set(h))
        w = list(set(w))
        n.sort()
        c.sort()
        h.sort()
        w.sort()
        print('count: ', duplicate_tensor_count)
        print('n: ', n)
        print('c: ', c)
        print('h: ', h)
        print('w: ', w)


if __name__ == '__main__':
    g = Generator()
    m = g.create_module()
    p = Parameters.from_list([[[1, 58, 28, 28], [1, 58, 28, 28]], 1])
    in_t = g.create_input(p)
    # print(in_t)
    t = torch.cat(in_t[0], in_t[1])
    print(t.shape)
    t = m.forward(in_t)
    # t = m.forward(in_t)
    # t = m.forward(in_t)
    print(t.shape)
