import torch
from torch import nn as nn


class Parser:
    @staticmethod
    def parse_hardtanh(params: list):
        # TODO - assumption: Hardtanh_ is always inplace, so inplace=True. The two inputs are min, max values
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                min_val = float(params[1])
                max_val = float(params[2])
                self.kernel = nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=True)

            def forward(self, x):
                return self.kernel(x)

        module = CustomModule(params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_unsqueeze(kernel_params: list):
        """ aten::slice|[1, 3, 299, 299]|0|0|9.22337e+18|1|
            aten::select|[1, 3, 299, 299]|1|2|
            aten::unsqueeze|[1, 299, 299]|1|

            This pattern is invoked by the interpreter by slicing the input to the squeeze function as we do here in
            the forward method.
        """

        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.dim = int(kernel_params[1])
                self.kernel = torch.unsqueeze

            def forward(self, x):
                return self.kernel(x[:, 0], self.dim)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_transpose(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.dim0 = int(kernel_params[1])
                self.dim1 = int(kernel_params[2])
                self.kernel = torch.transpose

            def forward(self, x):
                return self.kernel(x, self.dim0, self.dim1)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_dropout(kernel_params: list, in_place):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                p = float(_params[1])
                self.kernel = nn.Dropout(p=p, inplace=in_place)

            def forward(self, x):
                return self.kernel(x)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_flatten(kernel_params: list):
        class CustomModule(nn.Module):

            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.kernel = torch.flatten
                self.start_dim = int(_params[1])
                self.end_dim = int(_params[2])

            def forward(self, x):
                return self.kernel(x)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_chunk(kernel_params: list):
        # Chunk only exists in Shufflenet, in graphs where the compiler knows the div and dim inputs, it is represented
        # as prim::ConstantChunk. It always divides by 2 along dim 1 (in Shufflenet). Parsing ConstantChunk inputs,
        # gives us the input tensor size.
        class CustomModule(nn.Module):
            def __init__(self, _):
                super(CustomModule, self).__init__()
                self.kernel = torch.chunk

            def forward(self, x):
                return self.kernel(x, 2, 1)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_mul(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.scalar = float(_params[1])
                self.kernel = torch.mul

            def forward(self, x):
                return self.kernel(x, self.scalar)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_avg_pool2d(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()

                self.kernel = nn.AvgPool2d(kernel_size=_params[1], stride=_params[2], padding=_params[3],
                                           ceil_mode=bool(_params[4]), count_include_pad=bool(_params[5]),
                                           divisor_override=_params[6])

            def forward(self, x):
                return self.kernel(x)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_contiguous(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.format = int(_params[1])

            def forward(self, x: torch.Tensor):
                return x.contiguous(memory_format=self.format)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_mean(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.dim = _params[1]
                self.keep_dim = bool(_params[2])
                self.out = _params[3]
                self.kernel = torch.mean

            def forward(self, x):
                return self.kernel(x, self.dim, self.keep_dim)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_view(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.shape = kernel_params[1]

            def forward(self, x: torch.Tensor):
                return x.view(self.shape)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_reshape(kernel_params: list):
        # Some, if not most, of reshape calls are related to FusionGroup preparation. #TODO: in the future, when we
        # handle FusionGroup, we might remove this.
        class CustomModule(nn.Module):
            def __init__(self, _params):
                super(CustomModule, self).__init__()
                self.new_shape = _params[1]
                self.kernel = torch.reshape

            def forward(self, x: torch.Tensor):
                return self.kernel(x, self.new_shape)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_size(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _):
                super(CustomModule, self).__init__()

            def forward(self, x: torch.Tensor):
                return x.size()

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_t(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _):
                super(CustomModule, self).__init__()
                self.kernel = torch.t

            def forward(self, x):
                return self.kernel(x)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module

    @staticmethod
    def parse_div(kernel_params: list):
        class CustomModule(nn.Module):
            def __init__(self, _):
                super(CustomModule, self).__init__()
                self.kernel = torch.div

            def forward(self, x, y):
                return self.kernel(x, y)

        module = CustomModule(kernel_params)
        module = torch.jit.script(module)
        module = module.eval()
        module = module.cuda()
        return module
