from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.kernels.parameters.factory import get_parameters


class KernelRuntime:
    def __init__(self, kernel_name, kernel_parameters: BaseParameters = None):
        if kernel_parameters is None:
            self.kernel_parameters = get_parameters(kernel_name)
        else:
            self.kernel_parameters = kernel_parameters

        self.runtime = list()
        self.meta_data = None

    def store_runtime_data(self, runtime):
        self.runtime = runtime

    def store_meta_data(self, meta_data):
        self.meta_data = meta_data


if __name__ == '__main__':
    tmp = KernelRuntime("conv2d")
    print(tmp.kernel_parameters.from_list([[1, 192, 56, 56], [48, 192, 3, 3], None, [1, 1], [1, 1], [1, 1], 1]))
