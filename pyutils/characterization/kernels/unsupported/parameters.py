from pyutils.characterization.common.parameters import BaseParameters
from pyutils.common.strings import S_PARAMETERS, S_NAME
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION


class UnSupportedKernelParameters(BaseParameters):

    def __init__(self, parameters: list, name='unsupported_kernel'):
        # TODO - Assumption: only hardtanh_ happens inplace in the unsupported kernels
        # TODO: better handling of unsupported inplace kernels.
        super().__init__(name.rstrip('_'))
        self.parameters = parameters
        # self.inplace = True if str(name).endswith("_") else False

    def __str__(self):
        str_form = ""
        str_form += str(self.name) + "-"
        str_form += str(self.parameters)
        return str_form

    def to_dict(self):
        output = dict()
        output[S_NAME] = self.name
        output[S_PARAMETERS] = self.parameters

        return output

    def to_id(self):
        return self.to_list()

    def to_csv(self, delimiter=' '):
        input_shape = self.parameters[0]
        if type(input_shape) != list:
            input_shape = [input_shape]

        if len(self.parameters) > 1:
            rest_of_parameters = self.parameters[1:]
        else:
            rest_of_parameters = []

        obj = ''
        for c in input_shape:
            obj += str(c)
            obj += delimiter

        for c in rest_of_parameters:
            obj += str(c)
            obj += delimiter

        return obj[0:len(obj) - 1]

    @classmethod
    def from_dict(cls, parameters: dict):
        kernel_parameters = cls(parameters.get(S_PARAMETERS), parameters.get(S_NAME))
        kernel_parameters.generator_version = parameters.get(S_PARAMETERS_GENERATOR_VERSION, 1)
        return kernel_parameters
