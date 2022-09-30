import importlib
import os
import pkgutil

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters

kernels_version = "v2"
parent_package = "pyutils.characterization.kernels.parameters"


def _get_attr(kernel: str, attr: str):
    try:
        m = importlib.import_module(f"{parent_package}.{kernels_version}.{kernel}")
        attr = getattr(m, attr)
        return attr
    except ModuleNotFoundError:
        # Special handling for in-place kernels.
        m = importlib.import_module(f"{parent_package}.{kernels_version}.{kernel.rstrip('_')}")
        attr = getattr(m, attr)
        return attr
    except AttributeError:
        raise AttributeError(f"Could not find attribute {attr} for {kernel}")


def get_parameters(kernel: str) -> BaseParameters:
    return _get_attr(kernel, "Parameters")


def get_generator(kernel: str) -> BaseGenerator:
    return _get_attr(kernel, "Generator")


def get_module(kernel: str) -> BaseModule:
    return _get_attr(kernel, "Module")


def get_kernel_name(kernel: str):
    return _get_attr(kernel, "kernel_name")


def get_all_modules():
    v = importlib.import_module(f"{parent_package}.{kernels_version}")
    pkg_path = os.path.dirname(v.__file__)
    modules = set([name for _, name, _ in pkgutil.iter_modules([pkg_path])])
    return list(modules)


if __name__ == '__main__':
    print(get_parameters("conv2d"))
    print(get_generator("conv2d"))
    print(get_module("conv2d"))
