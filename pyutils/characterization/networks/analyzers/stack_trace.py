import ast
import logging
import os
from typing import List, Dict

import torch

from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.kernels.parameters import factory
from pyutils.characterization.kernels.unsupported.parameters import UnSupportedKernelParameters
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.networks.properties import Generator, get_torchvision_model
from pyutils.characterization.networks.utils import get_nw_stack_trace_file_path, get_unified_benchmark_name, \
    get_nw_last_executed_graph_file_path
from pyutils.common.config import JIT_FUSER
from pyutils.common.utils import FileUtils, GlobalLogger, prepare

logger = GlobalLogger(_logging_level=logging.INFO).get_logger()

DELIMITER = '|'


def create_stack_file(nw: str) -> str:
    """ Creates a PyTorch interpreter VM stack trace from scratch for a specific network.

    The generated file contains all the operations called/executed in the VM during the inference.
    It contains multiple executions, and we are interested in the last one which is the most optimized.
    This is why we call the inference several times. The generated file has marks for these different
    executions, which allow us to extract the last one.

    Parameters:
        nw: str, required
            Network name to create stack file for.

    Returns:
        f: Created stack trace file path.

    """
    logger.info(f"Creating new stack trace file for {nw} ...")
    f = get_nw_stack_trace_file_path(nw)
    FileUtils.silent_remove(f)
    prepare(f)

    nw = get_unified_benchmark_name(nw)

    generator = Generator(model_name=nw)
    params = generator.generate_random_input_parameters()
    in_t = generator.create_input(params).input_t
    model = get_torchvision_model(nw)(pretrained=True).cuda().eval()

    os.environ["CUSTOM_REPORT_STACK"] = "1"
    with torch.jit.fuser(JIT_FUSER):
        with torch.no_grad():
            with torch.jit.optimized_execution(True):
                model = torch.jit.trace(model, in_t)
                model = torch.jit.freeze(model)
                model = torch.jit.optimize_for_inference(model)
                # 4 was chosen to make sure that the last executed graph is truly the most
                # optimized. For Densenet, 2 iterations was not enough, fusing the BatchNorm
                # was done only at 4 iterations (Revisit this).
                for _ in range(4):
                    model(in_t)
                graph = torch.jit.last_executed_optimized_graph()
                num_of_ops = 0
                for node in graph.nodes():
                    if node.schema() == '(no schema)':
                        continue
                    else:
                        num_of_ops += 1
                unique_ops = torch.jit.export_opnames(model)
    os.environ["CUSTOM_REPORT_STACK"] = "0"

    if os.path.exists("stack_content.csv"):
        logger.info(f"Saving stack trace file to: {f}")
        logger.info(f"Number of operations: {num_of_ops}, unique ops: {unique_ops}")
        os.rename("stack_content.csv", f)
        content = f"End of stack content\n" + \
                  f"Number of nodes with schema: {num_of_ops}\n" + \
                  f"Number of unique operations: {len(unique_ops)}\n"
        FileUtils.serialize(content, f, append=True)
    else:
        raise Exception(f"Stack file for {nw} was not created.")
    return f


def read_stack_file(nw: str, overwrite=False):
    """ Reads the raw contents of the stack trace file if it exists, otherwise, create it first."""
    nw = get_unified_benchmark_name(nw)
    stack_f_path = get_nw_stack_trace_file_path(nw)
    if not os.path.exists(stack_f_path) or overwrite:
        stack_f_path = create_stack_file(nw)
    else:
        logger.info(f"Loading cached stack trace file at {stack_f_path}")
    content = FileUtils.deserialize(stack_f_path, return_type='raw')
    return content


def get_last_executed_graph_stack_trace(nw: str, overwrite_stack_file=False):
    """
    This method parses (and creates if non-existent) the stack trace of the specific network. Then, it
    filters out the content to extract the last executed graph (supposedly the most optimised).
    Args:
        nw: str, required
            The network string ID.
        overwrite_stack_file: bool, optional
            Whether to overwrite both the stack content file and the last executed graph stack file or not.

    Returns:
        content: list
            A list of kernels and their parameters that are executed in the last graph call.

    """

    def get_last_executed_optimized_graph_details(_content: list):
        """
        Reads the raw content of the stack trace CSV file and extracts the lines that include the last executed
        optimized graph.

        Args:
            _content: list, required
                The raw content of the file as lines.

        Returns:
            _content: list
                The extracted lines representing the operations of the last executed optimized graph.
            _num_num_of_graph_nodes: int
                The number of nodes in the graph.
            _num_num_of_unique_graph_ops: int
                The number of unique operations in the optimized graph

        """
        _num_of_graph_nodes = 0
        _num_of_unique_graph_ops = 0
        end = len(_content)
        for i in range(len(_content) - 1, 0, -1):
            l = _content[i]
            if l.__contains__("Number of nodes with schema"):
                _num_of_graph_nodes = int(l.split(":")[1].replace("\n", ""))
            if l.__contains__("Number of unique operations"):
                _num_of_unique_graph_ops = int(l.split(":")[1].replace("\n", ""))
            if l.__contains__("End of stack content"):
                end = i
            if l.__contains__("Stack tracing started"):
                _content = _content[i + 1:end]
                break
        return _content, _num_of_graph_nodes, _num_of_unique_graph_ops

    last_executed_graph_file_path = get_nw_last_executed_graph_file_path(nw)
    if not overwrite_stack_file and os.path.exists(last_executed_graph_file_path):
        logger.info("Loading cached last executed graph stack trace.")
        return FileUtils.deserialize(last_executed_graph_file_path)

    content = read_stack_file(nw, overwrite_stack_file)
    content, num_graph_nodes, num_unique_graph_nodes = get_last_executed_optimized_graph_details(content)

    # Save the last executed graph contents to a separate file.
    FileUtils.serialize(content, last_executed_graph_file_path, append=False)
    return content


def parse_kernel_from_stack_trace_line(line: str):
    """
    Given a line from the stack trace file of a specific network, this method extracts the kernel name and
    parameters list"
    Args:
        line: str, required
            A single line from the stack trace file.

    Returns:
        kernel_name: str
        kernel_parameters: List

    """

    def evaluate_str_parameters(tmp: List[str]):
        params = list()
        for i in range(len(tmp)):
            if tmp[i] == "NoneType":
                params.append(None)
            elif tmp[i] == '':
                continue
            else:
                params.append(ast.literal_eval(tmp[i]))
        return params

    line = line.replace('aten::', '')
    line = line.replace('prim::ConstantChunk', 'chunk')
    kernel_name = line.split(DELIMITER)[0]
    line = line.replace(kernel_name, '')
    line = list(line.lstrip(DELIMITER).rstrip(DELIMITER).split(DELIMITER))
    try:
        kernel_parameters = evaluate_str_parameters(line)
    except ValueError:
        if kernel_name != "div":
            # Div has a weird format, we don't parse it but it would generate too many warnings if not handled
            # e.g., aten::div|[]|[]|str|
            logger.warning(f"{kernel_name} parameters could not be evaluated correctly: {line}")
        return kernel_name, line
    return kernel_name, kernel_parameters


def parse_operations_from_stack_file(nw: str, overwrite_stack_file=False):
    """
    Parses the PyTorch interpreter VM operations of the last executed (i.e., most optimized) graph,
    from the stack file.

    Then processes the operations names to remove aten prefix, split and format the operations
    parameters. Finally, it creates a map of key values where the key is the operation name, and the value is a list
    of all parameters (as string) that were encountered in the stack trace file.

    Returns:
        operations: dict
            A dict object where the key is kernel name, and value is a list of parameters (each as a string).

    """
    content = get_last_executed_graph_stack_trace(nw, overwrite_stack_file)
    operations = dict()
    for record in content:
        kernel_name, kernel_parameters = parse_kernel_from_stack_trace_line(record)
        operations.setdefault(kernel_name, []).append(kernel_parameters)
    return operations


def parse_supported_kernel_parameters(kernel_name: str, parameters: list):
    """ For a particular kernel, this method first gets its Parameters class (if supported), then it uses it to
    parse the input parameters list and to create an instance of its Parameters class that is returned to the caller.
    """
    kernel_name = KernelsChecks.get_unified_kernel_name(kernel_name)
    if KernelsChecks.is_supported(kernel_name):
        params = factory.get_parameters(kernel_name).from_list(parameters)
    else:
        logger.warning(f"Unsupported kernel {kernel_name}")
        return None, None
    return params


def parse_unsupported_kernel_parameters(kernel_name: str, parameters: list):
    """
    This method parses an unsupported kernel parameters list and returns an object that can be used as a
    representation later for timing.
    """
    if KernelsChecks.is_unsupported(kernel_name):
        return UnSupportedKernelParameters(parameters, name=kernel_name)
    else:
        logger.warning(f"{kernel_name} is not handled.")
        return None


def parse_kernel_parameters_from_list(kernel_name: str, params: list):
    """
        This method parses a generic kernel parameters list and returns an object that can be used as a
        representation later for timing.
    """
    if KernelsChecks.is_supported(kernel_name):
        kernel_parameters = parse_supported_kernel_parameters(kernel_name, params)
    elif KernelsChecks.is_unsupported(kernel_name):
        kernel_parameters = parse_unsupported_kernel_parameters(kernel_name, params)
    elif KernelsChecks.is_unhandled(kernel_name):
        return None
    else:
        raise Exception(f"Unknown kernel category: {kernel_name}:{params}")
    return kernel_parameters


def get_kernels(nw: str, supported_only=False) -> Dict[str, List[BaseParameters]]:
    """
    Acts as an entry method to this module. It parses the PyTorch Interpreter VM operations as a map, then
     for each operation creates a map of kernel to a list of Parameters objects.

    Args:
        nw: str, required
            Network to parse the kernels for.

        supported_only: bool, optional
            Specifies whether we want to get all the kernels called in the network inference or just the supported ones
            (default: False)

    Returns:
        kernels: dict
            A map for the kernels that are run in @nw where the key is the kernel unified name, and the value is its
            parameters object list.

    """
    # Create a kernel to kernel parameters list map
    operations = parse_operations_from_stack_file(nw)

    kernels = dict()
    for name in operations.keys():
        for params in operations.get(name):
            if KernelsChecks.is_unhandled(name):
                continue
            if supported_only and not KernelsChecks.is_supported(name):
                continue
            kernel_parameters_obj: BaseParameters = parse_kernel_parameters_from_list(name, params)
            kernels.setdefault(name, []).append(kernel_parameters_obj)

    return kernels


def get_unique_kernels(nw: str, supported_only=False):
    kernels = get_kernels(nw, supported_only)
    unique_kernels = dict()
    for kernel_name, kernel_params in kernels.items():
        tmp_unique_set = set()
        for single_param in kernel_params:
            str_representation = str(single_param)
            if str_representation not in tmp_unique_set:
                tmp_unique_set.add(str_representation)
                unique_kernels.setdefault(KernelsChecks.get_unified_kernel_name(kernel_name), []).append(single_param)
    return unique_kernels


if __name__ == '__main__':
    pass
