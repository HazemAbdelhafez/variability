from abc import ABC
from typing import List, Union

from pyutils.characterization.common.types import TorchModuleParameters
from pyutils.common.arguments import BenchmarkGenericArgs, BlockBasedMeasurementArgs, \
    TorchModuleMeasurementArgs, is_graph_based
from pyutils.common.experiments_utils import logger
from pyutils.common.methods import is_power, is_runtime
from pyutils.common.strings import S_PWR_MEASUREMENT_METHOD, S_RUNTIME_MS, S_METRIC, S_BENCHMARK, \
    S_EXP_NOTES, \
    S_WARMUP_ITERATIONS, S_PWR_READINGS, S_DVFS_CONFIG, S_PWR_SMPLG_RATE, S_INTERNAL_PWR_METER, S_TIMING_METHOD, \
    S_CORE_AFFINITY, S_NUM_ISOLATED_CORES, S_NUM_THREADS, S_CONTROL_SCENARIO, S_NUM_OBSERVATIONS, S_BLOCK_RUNTIME_MS, \
    S_BLOCK_SIZE, S_RME, S_CONFIDENCE_LVL, S_RECORD_DATA, S_BATCH_SIZE, \
    S_PWR_EXP_TIME_SEC, S_PWR_METER, S_SUBMISSION_TIMESTAMP, S_NUM_RESTARTS, S_RESTART_ID, S_GRAPH_SIZE, S_CHECK_RME, \
    S_PARAMETERS
from pyutils.common.utils import FileUtils
from pyutils.hosts import agx


class ExpDataManager:
    def __init__(self):
        self.data = dict()

    # Tag: sample, before, after ... an ID to identify the temp sample
    def collect_temp(self, tag='sample'):
        if 'temperature' not in self.data.keys():
            self.data['temperature'] = dict()
        self.data['temperature'][tag] = agx.ThermalMonitor.get_temp_all()

    def parse_args(self, args):
        """ Parse default settings/meta-data from command line args dict, or configuration dict. They are parsed
        mostly as is. """
        raise NotImplementedError

    def save_dvfs_config(self, config):
        self.data[S_DVFS_CONFIG] = config

    def save_extra_config(self, config: dict):
        self.data.update(config)

    def save_metric_data(self, data: Union[dict, List]):
        raise NotImplementedError

    def write_to_disk(self, output_file_path):
        logger.info(f"Saving output to {output_file_path}")
        FileUtils.serialize(user_data=self.data,
                            file_path=output_file_path, append=True)

    def clean_up(self):
        pass


class CharacterizationExpDataManager(ExpDataManager, ABC):

    def parse_args(self, args: Union[BenchmarkGenericArgs, BlockBasedMeasurementArgs]):
        self.data[S_BENCHMARK] = args.bm
        self.data[S_WARMUP_ITERATIONS] = args.warmup_itrs
        self.data[S_NUM_OBSERVATIONS] = args.num_observations
        self.data[S_SUBMISSION_TIMESTAMP] = args.submission_ts
        self.data[S_CONTROL_SCENARIO] = args.control_scenario
        self.data[S_CORE_AFFINITY] = args.core_affinity
        self.data[S_NUM_ISOLATED_CORES] = args.num_isolated_cores
        self.data[S_NUM_THREADS] = args.num_threads
        self.data[S_RECORD_DATA] = args.record_data
        self.data[S_EXP_NOTES] = args.experiment_notes
        self.data[S_NUM_RESTARTS] = args.num_restarts
        self.data[S_RESTART_ID] = args.restart_id
        self.data[S_CHECK_RME] = args.check_rme

        self._parse_blockbased_args(args)

    def _parse_blockbased_args(self, args: BlockBasedMeasurementArgs):
        if isinstance(args, BlockBasedMeasurementArgs):
            self.data[S_RME] = args.rme
            self.data[S_CONFIDENCE_LVL] = args.confidence_lvl
            self.data[S_BLOCK_SIZE] = args.block_size
            self.data[S_BLOCK_RUNTIME_MS] = args.block_runtime_ms


class TorchModuleExpData(CharacterizationExpDataManager):

    def parse_args(self, args: TorchModuleMeasurementArgs):
        super().parse_args(args)
        self.data[S_METRIC] = args.metric
        self.data[S_BATCH_SIZE] = args.batch_size

        self._parse_pwr_exp_config(args)
        self._parse_runtime_exp_config(args)
        if is_graph_based(args):
            self.data[S_GRAPH_SIZE] = args.graph_size

    def _parse_pwr_exp_config(self, args):
        if is_power(args.metric):
            self.data[S_PWR_METER] = args.pwr_meter
            self.data[S_PWR_SMPLG_RATE] = args.sampling_rate
            self.data[S_PWR_MEASUREMENT_METHOD] = args.measurement_method

    def _parse_runtime_exp_config(self, args):
        if is_runtime(args.metric):
            self.data[S_TIMING_METHOD] = args.timing_method

    def save_module_parameters(self, parameters: TorchModuleParameters) -> None:
        self.data.update({S_PARAMETERS: parameters.to_dict()})

    def save_metric_data(self, data: Union[dict, List]):

        if is_runtime(self.data[S_METRIC]):
            self.data.update({S_RUNTIME_MS: data})

        if is_power(self.data[S_METRIC]):
            power_readings = dict()
            power_readings[S_PWR_READINGS] = dict()
            if self.data[S_PWR_METER] == S_INTERNAL_PWR_METER:
                power_readings[S_PWR_READINGS]['p_all'] = data['p_all']
                power_readings[S_PWR_READINGS]['p_cpu'] = data['p_cpu']
                power_readings[S_PWR_READINGS]['p_gpu'] = data['p_gpu']
                power_readings[S_PWR_READINGS]['p_soc'] = data['p_soc']
                power_readings[S_PWR_READINGS]['p_ddr'] = data['p_ddr']
                power_readings[S_PWR_READINGS]['p_sys'] = data['p_sys']
            else:
                power_readings[S_PWR_READINGS]['p_wattsup'] = data['p_wattsup']

            if 'duration' in data.keys():
                power_readings[S_PWR_EXP_TIME_SEC] = data['duration']
            self.data.update(power_readings)
