import torch

import pyutils.characterization.kernels.parameters.v2.cudnn_convolution_relu as cu_conv_relu
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.kernels.parameters.v2.cudnn_convolution_relu import Input

kernel_name = "conv2d"


class Names(cu_conv_relu.Names):
    pass


class Parameters(cu_conv_relu.Parameters):

    def __init__(self):
        super().__init__(kernel_name)


class Module(BaseModule):
    def __init__(self):
        super().__init__(name=kernel_name)
        self.kernel = torch.conv2d

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t, in_obj.weight_t, in_obj.bias,
                           in_obj.stride, in_obj.padding, in_obj.dilation, in_obj.groups)


class Generator(cu_conv_relu.Generator):
    def __init__(self):
        super().__init__()
        self.name = kernel_name
        self.module = Module()

    def generate_random_input_parameters(self):
        # TODO: re-implement this
        pass


if __name__ == "__main__":
    g = Generator()
    m = g.create_module()
    inp = g.create_input(Parameters.from_list([[1, 192, 56, 56], [48, 192, 3, 3], None, [1, 1], [1, 1], [1, 1], 1]))
    m.forward(inp)
    m.forward(inp)
    m.forward(inp)
    m.forward(inp)
    m.forward(inp)
