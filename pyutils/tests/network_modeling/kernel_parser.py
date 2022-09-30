import ast
import logging

import torch

from pyutils.common.utils import GlobalLogger

supported_kernels = ['aten::conv2d', 'aten::cudnn_convolution_relu', 'aten::matmul', "aten::_convolution",
                     "aten::relu_", "aten::max_pool2d", "aten::adaptive_avg_pool2d", "aten::linear", ]
unsupported_kernels = ["div", "Int", "view", "transpose", "contiguous", "prim::ConstantChunk", "size", "slice",
                       "select", "prim::TensorExprGroup"]

global_logger = GlobalLogger(logging.INFO)
logger = global_logger.get_logger()


def read_file(csv_f):
    r = list()
    with open(csv_f, 'r') as f:
        lines = f.readlines()
        # Find the last executed graph stack (i.e., most optimized)
        num_of_graph_nodes = 0
        num_of_unique_graph_ops = 0
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            l = lines[i]
            if l.__contains__("Number of nodes with schema"):
                num_of_graph_nodes = int(l.split(":")[1].replace("\n", ""))
            if l.__contains__("Number of unique operations"):
                num_of_unique_graph_ops = int(l.split(":")[1].replace("\n", ""))
            if l.__contains__("End of stack content"):
                end = i
            if l.__contains__("Stack tracing started"):
                lines = lines[i + 1:end]
                break
        # print(num_of_graph_nodes, num_of_unique_graph_ops)

        for l in lines:
            l = l.rstrip('\n')
            # l = l.replace('[', '(')
            # l = l.replace(']', ')')
            kernel, params = l.split('|')[0], l.split('|')[1:-1]

            print(kernel, params)
            kernel = kernel.replace('aten::', '')
            if kernel in unsupported_kernels:
                # Unhandled
                continue

            params_obj = list()
            for j in params:
                if j.__contains__("NoneType"):
                    params_obj.append(None)
                else:
                    params_obj.append(ast.literal_eval(j))

            if kernel == "size":
                tmp = torch.randn(params_obj[0], device='cuda')
                # Run size operation
                size_t = tmp.size(dim=params_obj[1])
                print(kernel, size_t, params_obj)
                continue

            try:
                kernel_obj = torch.__getattribute__(kernel)
            except AttributeError:
                try:
                    kernel_obj = torch.__getattribute__(f"_{kernel}")
                except AttributeError:
                    kernel_obj = torch.nn.functional.__getattribute__(kernel)

            if kernel_obj == torch.cudnn_convolution_relu or kernel_obj == torch.conv2d:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[1] = torch.randn(params_obj[1], device='cuda')
                if params_obj[2] is not None:
                    params_obj[2] = torch.randn(params_obj[2], device='cuda')
            if kernel_obj == torch.cudnn_convolution_add_relu:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')  # input
                params_obj[1] = torch.randn(params_obj[1], device='cuda')  # Weight
                params_obj[2] = torch.randn(params_obj[2], device='cuda')  # z
                params_obj[4] = torch.randn(params_obj[4], device='cuda')  # Bias

            if kernel_obj == torch.matmul:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[1] = torch.randn(params_obj[1], device='cuda')
            if kernel_obj == torch._adaptive_avg_pool2d:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
            if kernel_obj == torch.max_pool2d:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[-1] = bool(params_obj[-1])
            if kernel_obj == torch.flatten:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
            if kernel_obj == torch.add:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[1] = torch.randn(params_obj[1], device='cuda')
                torch.add(params_obj[0], params_obj[1], alpha=params_obj[2])
                # print(kernel, [ast.literal_eval(i) for i in params])
                continue

            if kernel_obj == torch.relu_:
                # in-place version of relu
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
            if kernel_obj == torch.mean:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[2] = bool(params_obj[2])
                if params_obj[3] is None:
                    params_obj = params_obj[:-1]

            if kernel_obj == torch.cat:
                tmp = list()
                for i in params_obj[0]:
                    tmp.append(torch.randn(i, device='cuda'))
                params_obj[0] = tmp

            if kernel_obj == torch.nn.functional.avg_pool2d:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')
                params_obj[4] = bool(params_obj[4])
                params_obj[5] = bool(params_obj[5])

            if kernel_obj == torch.nn.functional.hardtanh_:
                params_obj[0] = torch.randn(params_obj[0], device='cuda')

            kernel_obj(*params_obj)
            # # r.append((kernel_obj, params_obj))
            # print(kernel, [ast.literal_eval(i) for i in params])
            # print(kernel, [ast.literal_eval(i) for i in params])
            torch.batch_norm()
        return r
