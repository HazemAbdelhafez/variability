import torch

from pyutils.characterization.common.types import KernelParameters
from pyutils.characterization.kernels.parameters import factory
from pyutils.characterization.kernels.unsupported.parameters import UnSupportedKernelParameters
from pyutils.characterization.kernels.unsupported.parsers import Parser
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()


class KernelsLoaders:
    @staticmethod
    def load_unsupported_kernels(kernel_name, kernel_params):
        if kernel_name == 'hardtanh_' or kernel_name == "hardtanh":
            module = Parser.parse_hardtanh(kernel_params)
        elif kernel_name == 'unsqueeze':
            module = Parser.parse_unsqueeze(kernel_params)
        elif kernel_name == 'transpose':
            module = Parser.parse_transpose(kernel_params)
        elif kernel_name == 'view':
            module = Parser.parse_view(kernel_params)
        elif kernel_name == 'reshape':
            module = Parser.parse_reshape(kernel_params)
        elif kernel_name == 'size':
            module = Parser.parse_size(kernel_params)
        elif kernel_name in ['dropout', 'dropout_']:
            module = Parser.parse_dropout(kernel_params, kernel_name.__contains__('_'))
        elif kernel_name == 'flatten':
            module = Parser.parse_flatten(kernel_params)
        elif kernel_name == 'chunk':
            module = Parser.parse_chunk(kernel_params)
        elif kernel_name == 'contiguous':
            module = Parser.parse_contiguous(kernel_params)
        elif kernel_name == 'mul':
            module = Parser.parse_mul(kernel_params)
        elif kernel_name == 'avg_pool2d':
            module = Parser.parse_avg_pool2d(kernel_params)
        elif kernel_name == 'mean':
            module = Parser.parse_mean(kernel_params)
        elif kernel_name == 't':
            module = Parser.parse_t(kernel_params)
        else:
            logger.error(f"Kernel {kernel_name} is not supported. Aborting")
            raise AttributeError(f"{kernel_name} module doesn't exist.")

        return module

    @staticmethod
    def load_generator(kernel_name):
        if KernelsChecks.is_supported(kernel_name):
            generator = factory.get_generator(kernel_name)
            return generator
        else:
            logger.error(f"Kernel {kernel_name} is not supported. Aborting")
            raise AttributeError(f"{kernel_name} generator does not exist.")

    @staticmethod
    def load_kernel_parameters_parser(kernel_name):
        if KernelsChecks.is_supported(kernel_name):
            parameters_parser = factory.get_parameters(kernel_name).from_dict
            return parameters_parser
        elif KernelsChecks.is_unsupported(kernel_name):
            parameters_parser = UnSupportedKernelParameters.from_dict
            return parameters_parser
        else:
            logger.error(f"Kernel {kernel_name} is not supported. Aborting")
            raise AttributeError(f"{kernel_name} parameters do not exist.")

    @staticmethod
    def load_module_and_input(parameters: KernelParameters):
        kernel_name = parameters.name
        if KernelsChecks.is_supported(kernel_name):
            generator = KernelsLoaders.load_generator(kernel_name)()
            input_obj = generator.create_input(params=parameters)
            module = generator.create_module()

        elif KernelsChecks.is_unsupported(kernel_name):
            module = KernelsLoaders.load_unsupported_kernels(kernel_name, parameters.parameters)
            # TODO-Assumption: Unsupported kernels are hard-coded to run on the GPU
            input_obj = torch.randn(parameters.parameters[0]).cuda()
        else:
            return None, None
        return module, input_obj
