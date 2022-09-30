import torch

from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters


class BaseGenerator:
    def __init__(self, name="base"):
        self.generator_version = -1
        self.name = name
        self.module: BaseModule = BaseModule()

    def create_module(self):
        self.module = self.module.cuda()
        self.module = self.module.eval()
        self.module = torch.jit.script(self.module)
        return self.module

    def get_kernel(self):
        return self.module.kernel

    def create_input(self, params: BaseParameters, device: torch.device = 'cuda'):
        """
        Generates input from shape description
        """
        raise NotImplemented

    def generate_random_input_parameters(self):
        raise NotImplemented
