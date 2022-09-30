import torch


class BaseInput:
    @torch.jit.ignore
    def __init__(self):
        pass
