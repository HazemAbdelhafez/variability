import torch

from pyutils.characterization.common.inputs import BaseInput


class BaseModule(torch.nn.Module):
    def __init__(self, name="Base"):
        super(BaseModule, self).__init__()
        self.name = name
        self.kernel = None

    def forward(self, in_obj: BaseInput):
        raise NotImplemented
