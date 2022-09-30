from pyutils.characterization.kernels.parameters import factory
from pyutils.modeling.config import UNSUPPORTED_KERNELS, UNHANDLED_KERNELS


class KernelsChecks:
    @staticmethod
    def is_conv2d(kernel_name):
        return kernel_name in ['conv2d']

    @staticmethod
    def is_bn(kernel_name):
        return kernel_name in ['batchnorm2d', 'batch_norm']

    @staticmethod
    def is_maxpool(kernel_name):
        return kernel_name in ['max_pool2d', 'maxpool2d']

    @staticmethod
    def is_mm(kernel_name):
        return kernel_name in ['matmuladd', 'addmm', 'mm', 'matmul']

    @staticmethod
    def is_relu(kernel_name):
        return kernel_name in ['relu', 'relu_']

    @staticmethod
    def is_adaptivepool(kernel_name):
        return kernel_name in ['adaptivepool', 'adaptive_avg_pool2d', 'adaptivepool2d']

    @staticmethod
    def is_cat(kernel_name):
        return kernel_name in ['cat']

    @staticmethod
    def is_add(kernel_name):
        return kernel_name in ['add', 'add_']

    @staticmethod
    def is_fused(kernel_name):
        return kernel_name in ["cudnn_convolution_relu", "cudnn_convolution_add_relu"]

    @staticmethod
    def get_unified_kernel_name(kernel_name):
        if KernelsChecks.is_supported(kernel_name):
            return factory.get_kernel_name(kernel_name)
        else:
            return kernel_name.strip("_")

    @staticmethod
    def get_print_kernel_name(kernel_name):
        kernel_name = KernelsChecks.get_unified_kernel_name(kernel_name)

        if KernelsChecks.is_conv2d(kernel_name):
            return 'Conv2D'
        elif KernelsChecks.is_bn(kernel_name):
            return 'BatchNorm2D'
        elif KernelsChecks.is_maxpool(kernel_name):
            return 'MaxPool2D'
        elif KernelsChecks.is_mm(kernel_name):
            return 'Mat. Mul.'
        elif KernelsChecks.is_relu(kernel_name):
            return 'ReLU'
        elif KernelsChecks.is_adaptivepool(kernel_name):
            return 'AdaptivePool2D'
        elif KernelsChecks.is_cat(kernel_name):
            return 'Cat'
        elif KernelsChecks.is_add(kernel_name):
            return 'Add'
        else:
            return kernel_name.capitalize()

    @staticmethod
    def is_supported(kernel_name):
        return \
            KernelsChecks.is_conv2d(kernel_name) or \
            KernelsChecks.is_bn(kernel_name) or \
            KernelsChecks.is_maxpool(kernel_name) or \
            KernelsChecks.is_mm(kernel_name) or \
            KernelsChecks.is_add(kernel_name) or \
            KernelsChecks.is_adaptivepool(kernel_name) or \
            KernelsChecks.is_cat(kernel_name) or \
            KernelsChecks.is_relu(kernel_name) or \
            KernelsChecks.is_fused(kernel_name)

    @staticmethod
    def is_unsupported(kernel_name):
        unified_name = KernelsChecks.get_unified_kernel_name(kernel_name)
        return unified_name in [KernelsChecks.get_unified_kernel_name(i) for i in UNSUPPORTED_KERNELS]

    @staticmethod
    def is_unhandled(kernel_name):
        unified_name = KernelsChecks.get_unified_kernel_name(kernel_name)
        return unified_name in [KernelsChecks.get_unified_kernel_name(i) for i in UNHANDLED_KERNELS]

    @staticmethod
    def get_supported_kernels():
        kernels = list()
        all_modules = factory.get_all_modules()
        for module in all_modules:
            kernels.append(factory.get_parameters(module).name)
        return kernels


def is_kernel(kernel_name):
    return KernelsChecks.is_supported(kernel_name) or KernelsChecks.is_unsupported(kernel_name)
