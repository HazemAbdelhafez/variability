import os
from random import choice

import pyutils.characterization.networks.properties as nw_properties
from pyutils.hosts.common import HOSTNAME
from pyutils.characterization.common.types import TorchModuleParameters
from pyutils.characterization.kernels.utils.checks import is_kernel
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.characterization.networks.factory import NetworksLoaders
from pyutils.characterization.networks.utils import is_network
from pyutils.common.arguments import KernelGraphBasedMeasureTimeArgs, KernelBlockBasedMeasureTimeArgs, \
    NetworkGraphBasedMeasureTimeArgs, NetworkBlockBasedMeasureTimeArgs
from pyutils.common.arguments import TorchModuleMeasurementArgs
from pyutils.common.config import Timers, HOUR_TIMESTAMP
from pyutils.common.experiments_utils import Measurements
from pyutils.common.methods import join
from pyutils.common.paths import CHARACTERIZATION_DATA_DIR
from pyutils.common.strings import S_TIMING_METHOD, S_BENCHMARK, S_KERNEL_PARAMETERS, S_PARAMETERS
from pyutils.common.utils import FileUtils, prepare
from pyutils.common.utils import GlobalLogger
from pyutils.hosts.agx import DVFSHelpers

logger = GlobalLogger().get_logger()

# TODO-assumption: 2.5K is more than number of kernel calls in the biggest networks (e.g., DenseNet).
MAX_BLOCK_SIZE = 5000  # Maximum graph size in terms of kernel repetitions -


def pick(seq):
    return choice(seq)


def calculate_block_size(args: TorchModuleMeasurementArgs, current_block_runtime: float):
    overhead = Measurements.cuda_timer_overhead()
    # Estimate the target block size - faster than incrementing and trying again.
    min_runtime_to_discard_overhead = overhead / args.block_overhead_threshold
    target_block_runtime = max(min_runtime_to_discard_overhead, args.block_runtime_ms)

    new_block_size = int(args.block_size * target_block_runtime / current_block_runtime)
    # single_call_runtime = float(current_block_runtime / args.block_size)
    # logger.info(f"\nOverhead:                 {overhead} - "
    #             f"\nMin runtime:              {min_runtime_to_discard_overhead} - "
    #             f"\nSingle call runtime:      {single_call_runtime} - "
    #             f"\nInitial block runtime:    {current_block_runtime} - "
    #             f"\nTarget block runtime:     {target_block_runtime}")

    block_size = max(new_block_size, args.block_size)
    if block_size > MAX_BLOCK_SIZE:
        logger.warning(f"Estimated block size '{block_size}' is greater than the maximum limit '{MAX_BLOCK_SIZE}'. "
                       f"This is a low impact kernel (i.e., its runtime is much smaller than the timer overhead).")

        block_size = MAX_BLOCK_SIZE
    logger.info("Estimated new graph size: %d" % block_size)
    return block_size, overhead


def parse_measurement_args(config: dict) -> TorchModuleMeasurementArgs:
    if is_kernel(config[S_BENCHMARK]):
        return parse_kernel_measurement_args(config)
    if is_network(config[S_BENCHMARK]):
        return parse_network_measurement_args(config)


def parse_kernel_measurement_args(config: dict) -> TorchModuleMeasurementArgs:
    if config[S_TIMING_METHOD] == Timers.CUEvents.cuda_graphs:
        args = KernelGraphBasedMeasureTimeArgs(config)
    else:
        args = KernelBlockBasedMeasureTimeArgs(config)
    return args


def parse_network_measurement_args(config: dict) -> TorchModuleMeasurementArgs:
    if config[S_TIMING_METHOD] == Timers.CUEvents.cuda_graphs:
        args = NetworkGraphBasedMeasureTimeArgs(config)
    else:
        args = NetworkBlockBasedMeasureTimeArgs(config)
    return args


def parse_parameters(config: dict) -> TorchModuleParameters:
    if is_kernel(config[S_BENCHMARK]):
        return parse_kernel_parameters(config)
    if is_network(config[S_BENCHMARK]):
        return parse_network_parameters(config)


def parse_kernel_parameters(config: dict) -> TorchModuleParameters:
    return KernelsLoaders.load_kernel_parameters_parser(config[S_BENCHMARK])(config[S_KERNEL_PARAMETERS])


def parse_network_parameters(config: dict) -> TorchModuleParameters:
    return nw_properties.Parameters.from_dict(config[S_PARAMETERS])


def load_module_and_input(params: TorchModuleParameters):
    if is_kernel(params.name):
        return KernelsLoaders.load_module_and_input(params)
    if is_network(params.name):
        return NetworksLoaders.load_module_and_input(params)


def already_characterized(args: TorchModuleMeasurementArgs, params: str, dvfs_config: dict):
    data_summary_file = join(CHARACTERIZATION_DATA_DIR, "node-15", "networks", "kernels",
                             args.bm, "runtime", "characterization_data_summary.json")

    if not os.path.exists(data_summary_file):
        logger.info("Not characterized before: 1")
        return False

    contents = FileUtils.deserialize(data_summary_file)
    for i in contents:
        if params == i[S_PARAMETERS] and dvfs_config == DVFSHelpers.extract_dvfs_config(i):
            logger.info("Already characterized this config before.")
            logger.info(f"\n{params} - {dvfs_config}\n{i}")
            return True
    logger.info("Not characterized before: 2")
    return False


def get_output_file_path(args: TorchModuleMeasurementArgs):
    if is_kernel(args.bm):
        if args.characterize_children_kernels:
            return prepare(CHARACTERIZATION_DATA_DIR, HOSTNAME, "networks", "kernels", args.bm, args.metric,
                           f"raw_data_{HOUR_TIMESTAMP}.json")
    elif is_network(args.bm):
        return prepare(CHARACTERIZATION_DATA_DIR, HOSTNAME, args.bm, args.metric, f"raw_data_{HOUR_TIMESTAMP}.json")
    else:
        return prepare(CHARACTERIZATION_DATA_DIR, HOSTNAME, "tmp", args.bm, args.metric,
                       f"raw_data_{HOUR_TIMESTAMP}.json")
