import importlib
import os
from typing import List

import torch.nn.functional
from torch import Tensor

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.module import BaseModule
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.networks import utils
from pyutils.characterization.networks.utils import NetworksNameHelpers
from pyutils.common.modified_torchvision.models import tensor_based_inception_v3, tensor_based_googlenet
from pyutils.common.paths import SAVED_MODELS_DIR
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION
from pyutils.common.utils import prepare


def get_torchvision_model(model: str):
    try:
        if NetworksNameHelpers.is_googlenet(model):
            return tensor_based_googlenet
        elif NetworksNameHelpers.is_inception(model):
            return tensor_based_inception_v3
        else:
            m = importlib.import_module("torchvision.models")
            attr = getattr(m, model)
            return attr
    except AttributeError:
        raise AttributeError(f"Could not find attribute {model}")


def get_torchvision_model_input_shape(model: str, batch_size: int = 1):
    model = utils.get_unified_benchmark_name(model)
    if model in [utils.get_unified_benchmark_name(i) for i in
                 ["resnet", "densenet", "vgg", "googlenet", "shufflenet", "mobilenet", "mnasnet"]]:
        input_shape = [batch_size, 3, 224, 224]
    elif model == utils.get_unified_benchmark_name("alexnet"):
        input_shape = [batch_size, 3, 256, 256]
    elif model == utils.get_unified_benchmark_name("squeezenet"):
        input_shape = [batch_size, 3, 227, 227]
    elif model == utils.get_unified_benchmark_name("inception"):
        input_shape = [batch_size, 3, 299, 299]
    else:
        raise Exception(f"Unsupported network: {model}")
    return input_shape


class Names:
    name = "name"
    input_t_shape = "input_t_shape"
    torch_script_mode = "torch_script_mode"


class Parameters(BaseParameters):

    def __init__(self, model_name="Base", input_t_shape=None, torch_script_mode="tracing"):
        super().__init__(model_name)
        if input_t_shape is None:
            input_t_shape = [0, 0, 0, 0]
        self.input_t_shape = input_t_shape
        self.torch_script_mode = torch_script_mode

    def to_dict(self):
        output = dict()
        output[Names.name] = self.name
        output[Names.input_t_shape] = self.input_t_shape
        output[Names.torch_script_mode] = self.torch_script_mode
        output[S_PARAMETERS_GENERATOR_VERSION] = self.generator_version
        return output

    @classmethod
    def from_dict(cls, parameters):
        kernel_parameters = cls(parameters[Names.name],
                                parameters[Names.input_t_shape], parameters[Names.torch_script_mode])
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters

    @classmethod
    def from_list(cls, kernel_params: list):
        name = kernel_params[0]
        input_shape = kernel_params[1]
        torch_script_mode = kernel_params[2]
        return cls(name, input_shape, torch_script_mode)


class Input:
    @torch.jit.ignore
    def __init__(self, input_t_shape: List[int], device: torch.device = 'cuda'):
        self.input_t: Tensor = torch.randn(size=input_t_shape, device=device)


class Module(BaseModule):
    def __init__(self, model_name: str = "base",
                 mode: str = "tracing", batch_size: int = 1, device: torch.device = 'cuda'):
        super(Module, self).__init__(name=model_name)
        self.input_t_shape = get_torchvision_model_input_shape(model_name, batch_size)
        # TODO - assumption: input_t_shape always initializes before creating the module.
        self.kernel = self.create_module(mode, device)

    @torch.jit.ignore
    def create_module(self, mode: str = "tracing", device: torch.device = 'cuda'):
        # If the mode is cached, load it, else create it and save it
        model_path = prepare(SAVED_MODELS_DIR, f"{self.name}_{mode}.pt")
        if os.path.exists(model_path):
            # logger.info(f"Loading cached model from path: {model_path}...")
            model = torch.jit.load(model_path)
            # logger.info(f"Done.")
            return model
        else:
            model = get_torchvision_model(self.name)(pretrained=True).cuda().eval()
            if model is None:
                return model
            if mode == "scripting":
                model = torch.jit.script(model)
            else:
                in_t = Input(self.input_t_shape, device=device).input_t
                model = torch.jit.trace(model, in_t)
                model = torch.jit.freeze(model)
                model = torch.jit.optimize_for_inference(model)
                # Some networks required 4 runs to get the most optimized graph.
                for _ in range(4):
                    model(in_t)

            model.save(model_path)
            return model

    def forward(self, in_obj: Input):
        return self.kernel(in_obj.input_t)


class Generator(BaseGenerator):
    def __init__(self, model_name="base"):
        super().__init__(name=model_name)
        self.module = Module(model_name=model_name)

    def create_input(self, params: Parameters, device: torch.device = 'cuda'):
        return Input(params.input_t_shape, device)

    def create_module(self):
        return self.module

    def generate_random_input_parameters(self):
        return Parameters(self.name, self.module.input_t_shape)


class Tests:
    @staticmethod
    def test_forward():
        g = Generator()
        m = g.create_module()
        i = g.create_input(g.generate_random_input_parameters())
        print(m(i))


if __name__ == '__main__':
    pass
