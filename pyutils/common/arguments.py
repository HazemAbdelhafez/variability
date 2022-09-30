import os
from os.path import join as jp
from typing import Union

from pyutils.characterization.networks import utils
from pyutils.common.config import FRONT_IGNORED_POWER_RECORDS, Timers, TAIL_IGNORED_POWER_RECORDS, \
    IGNORED_TIMING_RECORDS, PowerMeasurementMethods
from pyutils.common.methods import parse_bool
from pyutils.common.paths import DVFS_CONFIGS_DIR
from pyutils.common.strings import S_NETWORK, S_PWR_MEASUREMENT_METHOD, S_RUNTIME, S_POWER, S_BENCHMARK, S_EXP_NOTES, \
    S_WARMUP_ITERATIONS, S_PWR_SMPLG_RATE, S_INTERNAL_PWR_METER, S_EXTERNAL_PWR_METER, S_CORE_AFFINITY, \
    S_NUM_ISOLATED_CORES, S_NUM_THREADS, S_CONTROL_SCENARIO, S_NUM_OBSERVATIONS, S_BLOCK_RUNTIME_MS, S_BLOCK_SIZE, \
    S_OUTPUT_DATA_FILE, S_RME, S_CONFIDENCE_LVL, S_RECORD_DATA, S_PROFILE_PWR, S_BATCH_SIZE, S_BM_ARGS, \
    S_DVFS_CONFIG_IDX, S_DVFS_CONFIG_FILE_PATH, S_PWR_METER, S_SUBMISSION_TIMESTAMP, S_SKIP_SEEN_BEFORE, \
    S_NUM_RESTARTS, S_RESTART_ID, S_DISCARDED_RESTARTS, S_GRAPH_SIZE, S_CHECK_RME, S_CHARACTERIZE_CHILDREN_KERNELS
from pyutils.hosts import agx


class BenchmarkGenericArgs:
    def __init__(self, args=None):
        self.bm = ''
        self.warmup_itrs = 5
        self.record_data = True
        self.dvfs_config_idx = 3149
        self.dvfs_config_file = jp(DVFS_CONFIGS_DIR, 'dvfs_455.json')
        self.submission_ts = '11111900'
        self.skip_seen_before = False
        self.control_scenario = 6  # Default
        self.num_threads = agx.AGX_CPU_CORE_COUNT
        self.num_observations = 50
        self.core_affinity = False
        self.num_isolated_cores = 0
        self.discarded_restarts = 0
        self.experiment_notes = ""
        self.num_restarts = 1
        self.restart_id = 0
        self.check_rme = True

        # This should always come last in the init method.
        if args is not None:
            self.parse_basic_args(args)

    def parse_basic_args(self, args):
        if S_NETWORK in args.keys():
            self.bm = str(args[S_NETWORK])
        elif S_BENCHMARK in args.keys():
            self.bm = str(args[S_BENCHMARK])
        else:
            pass
        self.bm = utils.get_unified_benchmark_name(self.bm)

        self.warmup_itrs = int(args[S_WARMUP_ITERATIONS]) if S_WARMUP_ITERATIONS in args.keys() else self.warmup_itrs
        self.record_data = parse_bool(args[S_RECORD_DATA]) if S_RECORD_DATA in args.keys() else self.record_data
        self.dvfs_config_idx = int(
            args[S_DVFS_CONFIG_IDX]) if S_DVFS_CONFIG_IDX in args.keys() else self.dvfs_config_idx

        # Set DVFS config file path with extra care.
        self.dvfs_config_file = str(
            args[S_DVFS_CONFIG_FILE_PATH]) if S_DVFS_CONFIG_FILE_PATH in args.keys() else self.dvfs_config_file
        if not os.path.exists(self.dvfs_config_file):
            self.dvfs_config_file = jp(DVFS_CONFIGS_DIR, self.dvfs_config_file)
            if not os.path.exists(self.dvfs_config_file):
                raise Exception(f"Configuration file path does not exit at: {self.dvfs_config_file}")

        self.submission_ts = str(
            args[S_SUBMISSION_TIMESTAMP]) if S_SUBMISSION_TIMESTAMP in args.keys() else self.submission_ts
        self.skip_seen_before = parse_bool(
            args[S_SKIP_SEEN_BEFORE]) if S_SKIP_SEEN_BEFORE in args.keys() else self.skip_seen_before

        self.control_scenario = int(
            args[S_CONTROL_SCENARIO]) if S_CONTROL_SCENARIO in args.keys() else self.control_scenario

        self.num_observations = int(
            args[S_NUM_OBSERVATIONS]) if S_NUM_OBSERVATIONS in args.keys() else self.num_observations

        self.core_affinity = parse_bool(args[S_CORE_AFFINITY]) if S_CORE_AFFINITY in args.keys() else self.core_affinity
        if S_NUM_ISOLATED_CORES in args.keys() and args[S_NUM_ISOLATED_CORES] is not None:
            self.num_isolated_cores = int(args[S_NUM_ISOLATED_CORES])

        self.num_threads = int(args[S_NUM_THREADS]) if S_NUM_THREADS in args.keys() else self.num_threads
        self.discarded_restarts = int(
            args[S_DISCARDED_RESTARTS]) if S_DISCARDED_RESTARTS in args.keys() else self.discarded_restarts
        self.experiment_notes = str(args[S_EXP_NOTES]) if S_EXP_NOTES in args.keys() else self.experiment_notes

        self.num_restarts = int(args[S_NUM_RESTARTS]) if S_NUM_RESTARTS in args.keys() else self.num_restarts
        self.restart_id = int(args[S_RESTART_ID]) if S_RESTART_ID in args.keys() else self.restart_id
        self.check_rme = bool(args[S_CHECK_RME]) if S_CHECK_RME in args.keys() else self.check_rme


class BlockBasedMeasurementArgs(BenchmarkGenericArgs):
    def __init__(self, args=None):
        super().__init__(args=args)
        self.rme = 0.5
        self.confidence_lvl = 99
        self.block_size = 100
        self.block_runtime_ms = 1000
        self.block_overhead_threshold = 0.001
        self.safety_cut_off_threshold = int(1e3)
        self.timing_method = Timers.CUEvents.block_based

        if args is not None:
            self.parse_block_based_measurement_args(args)

    def parse_block_based_measurement_args(self, args: dict):
        self.rme = float(args[S_RME]) if S_RME in args.keys() else self.rme
        self.confidence_lvl = float(args[S_CONFIDENCE_LVL]) if S_CONFIDENCE_LVL in args.keys() else self.confidence_lvl
        self.block_size = int(args[S_BLOCK_SIZE]) if S_BLOCK_SIZE in args.keys() else self.block_size
        self.block_runtime_ms = int(
            args[S_BLOCK_RUNTIME_MS]) if S_BLOCK_RUNTIME_MS in args.keys() else self.block_runtime_ms


class GraphBasedMeasurementArgs(BlockBasedMeasurementArgs):
    def __init__(self, args=None):
        super().__init__(args=args)
        self.graph_size = 1
        self.graph_replays = self.block_size  # An alias to block size.
        self.timing_method = Timers.CUEvents.cuda_graphs

        if args is not None:
            self.parse_graph_based_measurement_args(args)

    def parse_graph_based_measurement_args(self, args: dict):
        self.graph_size = int(args[S_GRAPH_SIZE]) if S_GRAPH_SIZE in args.keys() else self.graph_size


class MeasurePowerArgs(BenchmarkGenericArgs):
    def __init__(self, args=None):
        super().__init__(args)
        self.metric = S_POWER
        self.sampling_rate = 2
        # TODO: Remove this experiment time at some point. Not needed in block-based experiments.
        self.experiment_time_sec = 1
        self.measurement_method = PowerMeasurementMethods.block_based
        self.pwr_meter = S_INTERNAL_PWR_METER
        self.profile_pwr = False
        if args is not None:
            self.parse_measure_pwr_args(args)

    def parse_measure_pwr_args(self, args: dict):
        self.pwr_meter = str(args[S_PWR_METER]) if S_PWR_METER in args.keys() else self.pwr_meter
        if self.pwr_meter == S_EXTERNAL_PWR_METER:
            self.sampling_rate = 1  # External watts up meter supports maximum of 1 sample per second.
        self.sampling_rate = int(args[S_PWR_SMPLG_RATE]) if S_PWR_SMPLG_RATE in args.keys() else self.sampling_rate
        self.measurement_method = str(
            args[S_PWR_MEASUREMENT_METHOD]) if S_PWR_MEASUREMENT_METHOD in args.keys() else self.measurement_method
        self.profile_pwr = parse_bool(args[S_PROFILE_PWR]) if S_PROFILE_PWR in args.keys() else self.profile_pwr
        self.num_observations = self.num_observations + FRONT_IGNORED_POWER_RECORDS + TAIL_IGNORED_POWER_RECORDS


class NetworksBlockBasedMeasurePowerArgs(MeasurePowerArgs, BlockBasedMeasurementArgs):
    def __init__(self, args):
        MeasurePowerArgs.__init__(self, args)
        BlockBasedMeasurementArgs.__init__(self, args)
        self.batch_size = 1

        if args is not None:
            self.parse_block_based_measurement_args(args)
            self.parse_measure_pwr_args(args)

            # This line should always come last to overwrite any default values. It guarantees that one block is
            # enough to collect all power observations needed.
            self.block_runtime_ms = 1e3 * self.num_observations / self.sampling_rate
            self.batch_size = int(args[S_BATCH_SIZE]) if S_BATCH_SIZE in args.keys() else self.batch_size


class NetworkBlockBasedMeasureTimeArgs(BlockBasedMeasurementArgs):
    def __init__(self, args=None):
        BlockBasedMeasurementArgs.__init__(self, args)
        self.metric = S_RUNTIME
        self.num_observations = self.num_observations + IGNORED_TIMING_RECORDS
        self.batch_size = 1
        if args is not None:
            self.batch_size = int(args[S_BATCH_SIZE]) if S_BATCH_SIZE in args.keys() else self.batch_size


class NetworkGraphBasedMeasureTimeArgs(GraphBasedMeasurementArgs, NetworkBlockBasedMeasureTimeArgs):
    def __init__(self, args=None):
        GraphBasedMeasurementArgs.__init__(self, args)


class KernelBlockBasedMeasureTimeArgs(NetworkBlockBasedMeasureTimeArgs):
    def __init__(self, args=None):
        super().__init__(args)
        self.characterize_children_kernels = False if S_CHARACTERIZE_CHILDREN_KERNELS not in args.keys() else \
            bool(args[S_CHARACTERIZE_CHILDREN_KERNELS])


class KernelGraphBasedMeasureTimeArgs(GraphBasedMeasurementArgs, NetworkBlockBasedMeasureTimeArgs):
    def __init__(self, args=None):
        GraphBasedMeasurementArgs.__init__(self, args)
        self.characterize_children_kernels = False if S_CHARACTERIZE_CHILDREN_KERNELS not in args.keys() else \
            bool(args[S_CHARACTERIZE_CHILDREN_KERNELS])


class RodiniaBlockBasedMeasureTimeArgs(BlockBasedMeasurementArgs):
    def __init__(self, args=None):
        BlockBasedMeasurementArgs.__init__(self, args)
        self.metric = S_RUNTIME
        self.bm_args = ""
        self.timing_method = Timers.CUEvents.block_based
        self.num_observations = self.num_observations + IGNORED_TIMING_RECORDS
        self.output_data_file = '/tmp/tmp_data.json'
        if args is not None:
            self.bm_args = args[S_BM_ARGS]
            self.output_data_file = args[
                S_OUTPUT_DATA_FILE] if S_OUTPUT_DATA_FILE in args.keys() else self.output_data_file


class RodiniaBlockBasedMeasurePowerArgs(MeasurePowerArgs, BlockBasedMeasurementArgs):
    def __init__(self, args):
        MeasurePowerArgs.__init__(self, args)
        BlockBasedMeasurementArgs.__init__(self, args)
        self.bm_args = ""
        self.output_data_file = '/tmp/tmp_data.json'

        if args is not None:
            self.parse_block_based_measurement_args(args)
            self.parse_measure_pwr_args(args)

            # This line should always come last to overwrite any default values. It guarantees that one block is
            # enough to collect all power observations needed.
            self.block_runtime_ms = int(1e3 * self.num_observations / self.sampling_rate)
            self.output_data_file = args[
                S_OUTPUT_DATA_FILE] if S_OUTPUT_DATA_FILE in args.keys() else self.output_data_file

            self.bm_args = args[S_BM_ARGS]


# Type definitions
KernelMeasureTimeArgs = Union[KernelGraphBasedMeasureTimeArgs, KernelBlockBasedMeasureTimeArgs]
KernelMeasurePowerArgs = MeasurePowerArgs

NetworkMeasureTimeArgs = Union[NetworkGraphBasedMeasureTimeArgs, NetworkBlockBasedMeasureTimeArgs]
NetworkMeasurePowerArgs = Union[NetworkGraphBasedMeasureTimeArgs, NetworkBlockBasedMeasureTimeArgs]

TorchModuleMeasureTimeArgs = Union[KernelMeasureTimeArgs, NetworkMeasureTimeArgs]
TorchModuleMeasurePowerArgs = Union[NetworksBlockBasedMeasurePowerArgs, KernelMeasurePowerArgs]
TorchModuleMeasurementArgs = Union[TorchModuleMeasureTimeArgs, TorchModuleMeasurePowerArgs]


def is_graph_based(args):
    return isinstance(args, NetworkGraphBasedMeasureTimeArgs) or isinstance(args, KernelGraphBasedMeasureTimeArgs)


if __name__ == '__main__':
    t = NetworksBlockBasedMeasurePowerArgs(None)
    print(t.sampling_rate)
    print(t.block_size)
