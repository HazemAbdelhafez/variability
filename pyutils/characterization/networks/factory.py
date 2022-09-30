from pyutils.characterization.common.types import TorchModuleParameters
from pyutils.characterization.networks.properties import Generator, Parameters, get_torchvision_model_input_shape


class NetworksLoaders:
    @staticmethod
    def load_module_and_input(parameters: TorchModuleParameters):
        generator = Generator(model_name=parameters.name)
        input_obj = generator.create_input(parameters)
        module = generator.create_module()
        return module, input_obj

    @staticmethod
    def load_parameters(nw, batch_size=1):
        input_t_shape = get_torchvision_model_input_shape(nw, batch_size=batch_size)
        params = Parameters(model_name=nw, input_t_shape=input_t_shape)
        return params
