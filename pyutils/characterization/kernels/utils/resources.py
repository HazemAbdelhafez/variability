import os
from hashlib import sha256
from typing import List

from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.kernels.parameters import factory
from pyutils.characterization.networks.analyzers import stack_trace
from pyutils.characterization.networks.utils import VISION_NETWORKS
from pyutils.common.methods import join
from pyutils.common.paths import KERNELS_RESOURCES_DATA_DIR
from pyutils.common.utils import FileUtils


def get_unique_parameters(kernel: str, overwrite=False):
    """
    Extracts the unique input and kernel parameters from the torchvision models runs.

    Args:
        kernel: str, required
            The kernel to extract parameters for from the networks.

        overwrite: bool, optional
            Overwrite the previous unique parameters cached file if it exists (default: False)

    Returns:
        unique_params: list
            A list of Parameters objects for the unique occurrences of the target kernel in all VISION networks.

    Assumptions:
        We use VISION_NETWORKS as a fixed parameter in the search loop assuming that we are looking into these ones.

    """
    _parameters_class = factory.get_parameters(kernel)

    unique_parameters_file_path = join(KERNELS_RESOURCES_DATA_DIR, f"{_parameters_class.name}_unique_parameters.pkl")

    # Load if cached and not required to overwrite
    if os.path.exists(unique_parameters_file_path) and not overwrite:
        return FileUtils.deserialize(unique_parameters_file_path)
    else:
        FileUtils.silent_remove(unique_parameters_file_path)

    # Parse kernel parameters from all networks
    parameters: List[BaseParameters] = list()
    for nw in VISION_NETWORKS:
        kernels = stack_trace.get_kernels(nw)
        for i in kernels.get(kernel, []):
            parameters.append(i)

        # Special handling of in-place operations (e.g., ReLU)
        for i in kernels.get(f"{kernel}_", []):
            parameters.append(i)

    # Use hashing to extract unique parameters
    unique_params = list()
    tmp = set()
    for params in parameters:
        hash_code = sha256(params.__str__().encode('utf-8')).hexdigest()
        if tmp.__contains__(hash_code):
            continue
        else:
            tmp.add(hash_code)
            unique_params.append(params)

    # Save the unique parameters to a cached file.
    FileUtils.serialize(unique_params, unique_parameters_file_path)
    return unique_params


if __name__ == '__main__':
    print(get_unique_parameters(kernel="add"))
