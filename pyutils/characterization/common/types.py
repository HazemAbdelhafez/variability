from typing import Union

from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.kernels.unsupported.parameters import UnSupportedKernelParameters
from pyutils.characterization.networks.properties import Parameters as NWParameters

KernelParameters = Union[BaseParameters, UnSupportedKernelParameters]
NetworkParameters = Union[BaseParameters, NWParameters]

TorchModuleParameters = Union[KernelParameters, NetworkParameters]
