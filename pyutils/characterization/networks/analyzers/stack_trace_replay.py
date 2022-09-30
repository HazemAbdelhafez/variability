import os
from typing import Union

import torch

from pyutils.characterization.common.generator import BaseGenerator
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.kernels.unsupported.parameters import UnSupportedKernelParameters
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.characterization.networks.analyzers.stack_trace import get_last_executed_graph_stack_trace, \
    parse_kernel_from_stack_trace_line, parse_kernel_parameters_from_list
from pyutils.common.config import JIT_FUSER
from pyutils.common.methods import join
from pyutils.common.paths import STACK_TRACE_REPLAY_DIR
from pyutils.common.utils import GlobalLogger, FileUtils

logger = GlobalLogger().get_logger()


def replay_kernel(kernel_parameters: Union[BaseParameters, UnSupportedKernelParameters]):
    """
    Given a kernel_parameters object, this method replays it.
    Args:
        kernel_parameters: Union[BaseParameters, UnSupportedKernelParameters], required

    Returns:
        The output of the kernel execution.
    """
    kernel_name = kernel_parameters.name
    logger.info(f"Replaying: {kernel_name} : {kernel_parameters}")
    if KernelsChecks.is_supported(kernel_name):
        generator: BaseGenerator = KernelsLoaders.load_generator(kernel_name)()
        input_obj = generator.create_input(params=kernel_parameters)
        module = generator.create_module()

    elif KernelsChecks.is_unsupported(kernel_name):
        module = KernelsLoaders.load_unsupported_kernels(kernel_name, kernel_parameters.parameters)
        input_obj = torch.randn(kernel_parameters.parameters[0]).cuda()

    else:
        return None

    return module.forward(input_obj)


def replay_stack_trace(nw, overwrite_stack_file=False):
    """
    Given a network name, this method load's its last executed optimized graph, and replays its kernels.
    Args:
        nw:
        overwrite_stack_file:

    Returns:

    """
    last_executed_graph_stack = get_last_executed_graph_stack_trace(nw, overwrite_stack_file)
    for kernel_line in last_executed_graph_stack:
        kernel_name, kernel_parameters = parse_kernel_from_stack_trace_line(kernel_line)
        if KernelsChecks.is_unhandled(kernel_name):
            # logger.info(f"Ignoring {kernel_name} from the replay.")
            continue
        kernel_parameters = parse_kernel_parameters_from_list(kernel_name, kernel_parameters)
        replay_kernel(kernel_parameters)


def record_stack_trace(nw: str, overwrite_stack_file=False):
    """ This method does the following:
        1- Reads a network's stack trace file
        2- Extracts the last executed graph content
        3- Replays the kernels from the last graph
        4- Records the stack created from this replay.

        Args:
        nw: str, required
        overwrite_stack_file: bool, optional

        Returns:
            The output of the replayed stack.
    """
    f = join(STACK_TRACE_REPLAY_DIR, f"{nw}_stack_trace.txt")
    FileUtils.silent_remove(f)

    os.environ["CUSTOM_REPORT_STACK"] = "1"
    with torch.jit.fuser(JIT_FUSER):
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                replay_stack_trace(nw, overwrite_stack_file)
    os.environ["CUSTOM_REPORT_STACK"] = "0"

    if os.path.exists("stack_content.csv"):
        logger.info(f"Saving stack trace file to: {f}")
        os.rename("stack_content.csv", f)
    else:
        raise Exception(f"Stack file for {nw} was not created.")

    return FileUtils.deserialize(f)


if __name__ == '__main__':
    pass
