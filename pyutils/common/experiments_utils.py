import copy
import datetime
import os
from os.path import join as jp
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch.cuda

from pyutils.characterization.networks.utils import get_unified_benchmark_name
from pyutils.common.arguments import BenchmarkGenericArgs
from pyutils.common.config import DVFS_SPACE_SIZE, GENERIC_SPACE_SIZE, DUMMY_RESTARTS, RME_THRESHOLD, CONFIDENCE_LVL, \
    MIN_NUM_OF_VALID_RESTARTS, JIT_FUSER
from pyutils.common.paths import CHARACTERIZATION_DATA_DIR, DVFS_CONFIGS_DIR
from pyutils.common.strings import S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_CONTROL_SCENARIO, S_DVFS_CONFIG_IDX, \
    S_DVFS_CONFIG_FILE_PATH
from pyutils.common.timers import CudaStopWatch, StopWatch
from pyutils.common.utils import FileUtils, GlobalLogger, MHDMY_DATE_FORMAT
from pyutils.hosts import agx
from pyutils.hosts.agx import DVFSHelpers

logger = GlobalLogger().get_logger()


class ConfigHandler:
    def __init__(self, add_to_history=True,
                 generate_alternatives=True,
                 skip_seen_before=True,
                 prev_configs_file=None):
        self.data = ''
        self.add_to_history = add_to_history
        self.prev_configs_file = prev_configs_file
        self.max_count = DVFS_SPACE_SIZE * GENERIC_SPACE_SIZE
        self.generate_alternatives = generate_alternatives
        self.skip_seen_before = skip_seen_before

    def get_seen_before_configs(self, kernel_parameters):
        seen_before = set()
        with open(self.prev_configs_file, 'r') as file_obj:
            for line in file_obj.readlines():
                prev_combination = line
                if kernel_parameters is not None:
                    prev_combination = prev_combination.replace(' ', '_')
                prev_combination = prev_combination.replace('\n', '')
                seen_before.add(prev_combination)
        return seen_before

    def get_config_for_bm(self, dvfs_config, kernel_parameters):
        dvfs_config_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
        # Make sure we didn't pick this combination before
        seen_before = False
        if os.path.exists(self.prev_configs_file):
            configs_seen_before = self.get_seen_before_configs(kernel_parameters)
            # Create the lookup key
            lookup_key = dvfs_config_id + '_' + kernel_parameters.to_csv(delimiter='_')
            if configs_seen_before.__contains__(lookup_key):
                self.add_to_history = False
                if self.skip_seen_before:
                    logger.info(f"Config seen before: {lookup_key}. Skipping it.")
                    seen_before = True
                    logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
                    return dvfs_config, kernel_parameters, seen_before
                else:
                    logger.info(f"Config seen before: {lookup_key}. Yet we will run it as required.")
                    logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
                    return dvfs_config, kernel_parameters, seen_before

        self.data = f"{dvfs_config_id} {kernel_parameters.to_csv()}"
        logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
        return dvfs_config, kernel_parameters, seen_before

    def filter_seen_before(self, dvfs_config, kernel_parameters, prev_configs_file,
                           kernel_parameters_generator=None, dvfs_space_ratio=1.0, skip_if_seen=False):
        self.prev_configs_file = prev_configs_file
        dvfs_config_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
        # Make sure we didn't pick this combination before
        seen_before = False
        if os.path.exists(prev_configs_file):
            configs_seen_before = self.get_seen_before_configs(kernel_parameters)

            # Create the lookup key
            lookup_key = dvfs_config_id
            if kernel_parameters is not None:
                lookup_key += '_' + kernel_parameters.to_csv(delimiter='_')

            if configs_seen_before.__contains__(lookup_key):
                self.add_to_history = False
                if skip_if_seen:
                    logger.info(f"Config seen before: {lookup_key}. Skipping it.")
                    seen_before = True
                    logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
                    return dvfs_config, kernel_parameters, seen_before
                else:
                    logger.info(f"Config seen before: {lookup_key}. Yet we will run it as required.")
                    seen_before = False
                    logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
                    return dvfs_config, kernel_parameters, seen_before

            # This is a termination condition
            seen_before_count = len(configs_seen_before)
            while configs_seen_before.__contains__(lookup_key) and seen_before_count < self.max_count:
                dvfs_config = DVFSHelpers.generate_random_dvfs_config_partial(ratio=dvfs_space_ratio)
                dvfs_config_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
                lookup_key = dvfs_config_id
                if kernel_parameters_generator is not None:
                    kernel_parameters = kernel_parameters_generator.generate_random_input_parameters()
                    lookup_key += '_' + kernel_parameters.to_csv(delimiter='_')

        self.data = f"{dvfs_config_id}"
        if kernel_parameters is not None:
            self.data += " " + f"{kernel_parameters.to_csv()}"

        logger.info(f"-- DVFS config after filtering seen before: {dvfs_config}")
        return dvfs_config, kernel_parameters, seen_before

    def save_to_history(self):
        if self.add_to_history:
            logger.info("Saving configuration to history file after completing successfully")
            FileUtils.serialize(self.data, self.prev_configs_file, file_extension='txt', append=True)

    @staticmethod
    def seen_before(dvfs_config, kernel_parameters, prev_configs_file):
        dvfs_config_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
        # Make sure we didn't pick this combination before
        if os.path.exists(prev_configs_file):
            seen_before = set()
            seen_before_count = 0
            with open(prev_configs_file, 'r') as file_obj:
                for line in file_obj.readlines():
                    prev_combination = line
                    if kernel_parameters is not None:
                        prev_combination = prev_combination.replace(' ', '_')
                    prev_combination = prev_combination.replace('\n', '')
                    seen_before.add(prev_combination)
                    seen_before_count += 1

            # Create the lookup key
            if kernel_parameters is None:
                lookup_key = dvfs_config_id
            else:
                lookup_key = dvfs_config_id + '_' + kernel_parameters.to_csv(delimiter='_')

            if seen_before.__contains__(lookup_key):
                return True
        return False


def parse_dvfs_configs(target_file):
    return FileUtils.deserialize(jp(DVFS_CONFIGS_DIR, target_file), return_type='list')


def get_unique_submission_timestamps(bm, metric, num_ts=4, node='node-01'):
    p = jp(CHARACTERIZATION_DATA_DIR, f'{node}/{get_unified_benchmark_name(bm)}/'
                                      f'{metric}/raw_data_summary_with_timestamp.pickle')
    tss = set()

    df: pd.DataFrame = FileUtils.deserialize(p)
    c = list(df['submission_timestamp'].unique())
    for j in c:
        tss.add(datetime.datetime.strptime(j, MHDMY_DATE_FORMAT))
    lt = list(tss)
    lt.sort()
    if num_ts > len(lt):
        lt = [i.strftime(MHDMY_DATE_FORMAT) for i in lt]
    else:
        lt = [i.strftime(MHDMY_DATE_FORMAT) for i in lt][-num_ts:]
    logger.info(f"Unique timestamps for {bm}:{metric} -> {lt}")
    return lt


def node_id_to_int(node_id: str):
    return int(node_id.split('-')[1])


def ran_enough_restarts(restart_tmp_file):
    measurements = FileUtils.deserialize(restart_tmp_file)
    if len(measurements) < DUMMY_RESTARTS + MIN_NUM_OF_VALID_RESTARTS:
        return False

    measurements = [float(i) for i in measurements[DUMMY_RESTARTS:]]
    if Measurements.collected_enough_measurements(measurements, confidence_lvl=CONFIDENCE_LVL,
                                                  rme_threshold=RME_THRESHOLD):
        return True
    return False


def get_missing_configs(input_configs: list, reference_configs_input_file: str):
    """ Compares an input list of configs and a reference one, returns indexes of missing reference configs in the
    input list.
    @input_configs: a list of DVFS dictionaries.
    @reference_configs_input_file: reference configurations source file path
    @return dvfs_configs, and their indexes in a specified input_file"""

    reference_configs = FileUtils.deserialize(reference_configs_input_file)
    results = list()
    indexes = list()
    for i in range(len(reference_configs)):
        rc = reference_configs[i]
        found = False
        for ic in input_configs:
            if DVFSHelpers.is_equal(rc, ic):
                found = True
                break
        if not found:
            results.append(rc)
            indexes.append(i)
    return results, indexes


def get_missing_configs_from_df(input_configs_file, reference_configs_input_file: str):
    """ Compares an input list of configs and a reference one, returns indexes of missing reference configs in the
    input list.
    @input_configs: a list of DVFS dictionaries.
    @reference_configs_input_file: reference configurations source file path
    @return dvfs_configs, and their indexes in a specified input_file"""
    reference_configs_input_file = jp(DVFS_CONFIGS_DIR, reference_configs_input_file)
    reference_configs = FileUtils.deserialize(reference_configs_input_file)
    reference_df = pd.DataFrame(reference_configs)
    if isinstance(input_configs_file, str):
        input_configs_df = FileUtils.deserialize(input_configs_file)
    elif isinstance(input_configs_file, pd.DataFrame):
        input_configs_df = input_configs_file
    else:
        raise Exception("Unsupported type.")

    input_configs_df = input_configs_df.drop(
        columns=[i for i in input_configs_df.columns if i not in [S_CPU_FREQ, S_GPU_FREQ,
                                                                  S_MEMORY_FREQ]])
    res = pd.merge(reference_df, input_configs_df, how='outer', indicator=True, on=[S_CPU_FREQ, S_GPU_FREQ,
                                                                                    S_MEMORY_FREQ])
    res = res[res['_merge'] == 'left_only'].copy()
    res.drop(columns=['_merge'], inplace=True)
    results = res.to_dict(orient='records')
    indexes = list(res.index)
    return results, indexes


def get_dvfs_configs_indexes(input_configs_df: pd.DataFrame, reference_configs_input_file: str):
    """ This method takes a Dataframe and a reference DVFS config file. It matches the entries in
    both and returns the configs and their indexes that exist in the Dataframe and DVFS config file. """
    reference_configs_input_file = jp(DVFS_CONFIGS_DIR, reference_configs_input_file)
    reference_configs = FileUtils.deserialize(reference_configs_input_file)
    reference_df = pd.DataFrame(reference_configs)
    input_configs_df = input_configs_df.drop(columns=[i for i in input_configs_df.columns if i not in
                                                      [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ]])
    res = pd.merge(reference_df, input_configs_df, how='outer', indicator=True, on=[S_CPU_FREQ, S_GPU_FREQ,
                                                                                    S_MEMORY_FREQ])
    res = res[res['_merge'] == 'both'].copy()
    res.drop(columns=['_merge'], inplace=True)
    results = res.to_dict(orient='records')
    indexes = list(res.index)
    return results, indexes


class Measurements:

    @staticmethod
    def margin_of_error(array, confidence_level_percent=99):
        # We use z-score, so the sample size should at least be 30 or otherwise,
        # we should use T-score.
        # More info: https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals_print.html
        if len(array) < 30:
            print("For accurate relative margin of error, use bigger sample (>30).")

        # Standard error of mean
        se = np.std(array) / np.sqrt(len(array))

        # Margin of error
        if confidence_level_percent == 99:
            me = 2.576 * se  # 99% confidence
        elif confidence_level_percent == 95:
            me = 1.96 * se  # 95% confidence
        else:
            # Default is 95%
            me = 1.96 * se
        return me

    @staticmethod
    def ci_bounds(array, confidence_lvl=99):
        me = Measurements.margin_of_error(array, confidence_lvl)
        return np.mean(array) - me, np.mean(array) + me

    @staticmethod
    def relative_margin_of_error(array, confidence_level_percent=99):
        me = Measurements.margin_of_error(array, confidence_level_percent)
        # Relative margin of error
        rme = 100 * me / np.mean(array)
        return rme

    @staticmethod
    def collected_enough_measurements(arr, rme_threshold=0.25, min_count=31, confidence_lvl=95, metric='median',
                                      return_rme=False) -> Union[bool, Tuple[bool, float]]:
        # To avoid manipulating the order of the original data.
        arr = copy.copy(arr)

        if len(arr) < min_count:
            return False

        if metric == 'mean':
            rme = Measurements.relative_margin_of_error(arr, confidence_level_percent=confidence_lvl)
        else:
            rme = Measurements.median_relative_margin_of_error(arr, confidence_level_percent=confidence_lvl)

        status = False
        # Target relative margin of error threshold
        if rme <= rme_threshold:
            # We need a minimum number of samples, for later analysis, before aborting.
            status = True
        if return_rme:
            return status, rme
        else:
            return status

    @staticmethod
    def median_ci_bounds(arr, confidence_level_percent=95):
        lower, upper = Measurements.median_ci_bounds_indexes(arr, confidence_level_percent)
        return arr[lower], arr[upper]

    @staticmethod
    def median_ci_bounds_indexes(arr, confidence_level_percent=95):
        n = len(arr)
        arr.sort()
        z_score = 2.576 if confidence_level_percent == 99 else 1.96  # 95% otherwise
        lower = int(np.floor(n / 2 - z_score * np.sqrt(n) / 2))
        upper = int(np.ceil(1 + n / 2 + z_score * np.sqrt(n) / 2))
        return max(0, lower), min(n - 1, upper)

    @staticmethod
    def ci_for_differences(a_1, a_2, confidence_lvl=99, percentile=0.5):
        n = len(a_1)
        m = len(a_2)
        z_score = 2.576 if confidence_lvl == 99 else 1.96  # 95% otherwise

        k = int(np.ceil(n * m * percentile - z_score * (np.sqrt(n * m * (n + m + 1) / 12))))
        a_1 = np.sort(a_1)
        a_2 = np.sort(a_2)
        differences = list()
        for i in a_1:
            for j in a_2:
                differences.append(i - j)
        differences = np.sort(differences)
        # Sanity check
        if k < 0:
            k = 0
        if k >= n:
            k = n - 1
        return differences[k], differences[-k]

    @staticmethod
    def median_relative_margin_of_error(arr, confidence_level_percent=99):
        lower_ci, upper_ci = Measurements.median_ci_bounds(arr, confidence_level_percent)
        m = np.median(arr)
        lower_rme = 100 * abs(m - lower_ci) / m
        upper_rme = 100 * abs(upper_ci - m) / m
        return max(lower_rme, upper_rme)

    @staticmethod
    def cuda_timer_overhead(timer_obj: CudaStopWatch = None, n=50):
        s = torch.cuda.Stream(priority=-1)
        with torch.cuda.stream(s):
            if timer_obj is None:
                timer_obj = CudaStopWatch()
            result = np.zeros(n)
            for i in range(n):
                timer_obj.start()
                timer_obj.stop()
                torch.cuda.current_stream().synchronize()
                result[i] = timer_obj.elapsed_ms()
            torch.cuda.current_stream().synchronize()

        return np.median(result)

    @staticmethod
    def cuda_sync_overhead(n=100):
        timer = StopWatch()
        timer.start()
        for _ in range(n):
            torch.cuda.synchronize()
        timer.stop()
        return timer.elapsed_ms(n)


class VariabilityMetrics:
    @staticmethod
    def max_min_diff(arr):
        return max(arr) - min(arr)

    @staticmethod
    def quantiles_diff(arr, lower_quantile=.25, upper_quantile=.75):
        if upper_quantile < lower_quantile:
            raise Exception("Upper should be > lower quantile.")
        return np.quantile(arr, upper_quantile) - np.quantile(arr, lower_quantile)

    @staticmethod
    def cov(arr):
        # coefficient of variation: probably won't use it as opposed to Range and Median
        # which are better suited for non-parametric distributions.
        # https://arxiv.org/pdf/1907.01110.pdf
        return np.std(arr) / np.mean(arr)

    @staticmethod
    def iqr(arr):
        diff = VariabilityMetrics.quantiles_diff(arr, .25, .75)
        return diff

    @staticmethod
    def qcv(arr):
        """ Quartile-based coefficient of variation. """
        r = VariabilityMetrics.quantiles_diff(arr, .25, .75) / np.median(arr)
        return r

    @staticmethod
    def mad(arr):
        r = np.median(abs(np.array(arr) - np.median(arr)))
        return r

    @staticmethod
    def madm(arr):
        return VariabilityMetrics.mad(arr) / np.median(arr)

    @staticmethod
    def qcd(arr):
        """ Quartile coefficient of dispersion.
        This statistic has been suggested as a robust alternative to the coefficient of variation. """
        n = VariabilityMetrics.quantiles_diff(arr, .25, .75)
        d = np.quantile(arr, 0.25) + np.quantile(arr, 0.75)
        return n / d

    @staticmethod
    def calculate_all(arr, metrics=("madm",)):
        output = dict()
        if 'qcd' in metrics:
            output['qcd'] = 100 * VariabilityMetrics.qcd(arr)
        if 'qcv' in metrics:
            output['qcv'] = 100 * VariabilityMetrics.qcv(arr)
        # output['mad'] = VariabilityMetrics.mad(arr)
        if 'madm' in metrics:
            output['madm'] = 100 * VariabilityMetrics.madm(arr)
        if 'iqr' in metrics:
            output['iqr'] = VariabilityMetrics.iqr(arr)
        # output['custom'] = VariabilityMetrics.custom(arr)
        # output['cov'] = VariabilityMetrics.cov(arr)
        # output['std'] = np.std(arr)
        # output['max_min_diff'] = VariabilityMetrics.max_min_diff(arr)
        return output


class DVFSManager:
    def __init__(self, config):
        if type(config) is dict:
            self.dvfs_config_idx = int(config[S_DVFS_CONFIG_IDX]) if S_DVFS_CONFIG_IDX in config.keys() else 0
            self.dvfs_configs_file_path = str(config[S_DVFS_CONFIG_FILE_PATH]) \
                if S_DVFS_CONFIG_FILE_PATH in config.keys() else jp(DVFS_CONFIGS_DIR, 'dvfs_3150.json')
            self.control_scenario = 6 if S_CONTROL_SCENARIO not in config.keys() else int(config[S_CONTROL_SCENARIO])

        elif isinstance(config, BenchmarkGenericArgs):
            self.dvfs_config_idx = config.dvfs_config_idx
            self.dvfs_configs_file_path = config.dvfs_config_file
            self.control_scenario = config.control_scenario

        self.dvfs_config, _ = DVFSHelpers.get_a_dvfs_config(self.dvfs_config_idx, self.dvfs_configs_file_path)

    def setup(self):
        if self.control_scenario == 1:
            agx.BoardController.reset_board_to_maximum()
            self.dvfs_config = {S_CPU_FREQ: -1, S_GPU_FREQ: -1, S_MEMORY_FREQ: -1}

        elif self.control_scenario == 2:
            # Reset board to maximum settings
            agx.BoardController.reset_board_to_maximum()
            agx.CPUController.disable_railgate()
            agx.GPUController.disable_railgate()

            # Set both the GPU and CPU scaling governors to userspace
            agx.CPUController.set_scaling_governor('userspace')
            agx.GPUController.set_scaling_governor('userspace')

            # Set lower and upper bounds just to be sure
            agx.CPUController.set_freq_lower_bound(agx.CPUController.get_max_freq())
            agx.CPUController.set_freq_upper_bound(agx.CPUController.get_max_freq())
            agx.GPUController.set_freq_lower_bound(agx.GPUController.get_max_freq())
            agx.GPUController.set_freq_upper_bound(agx.GPUController.get_max_freq())

            # Set current frequency to max
            agx.CPUController.set_freq_to_max()
            agx.GPUController.set_freq_to_max()
            agx.MemoryController.set_freq_to_max()
            self.dvfs_config = {S_CPU_FREQ: agx.CPUController.get_max_freq(),
                                S_GPU_FREQ: agx.GPUController.get_max_freq(),
                                S_MEMORY_FREQ: agx.MemoryController.get_max_freq()}

        elif self.control_scenario in [3, 4, 5]:
            agx.BoardController.reset_dvfs_settings()
            self.dvfs_config = {S_CPU_FREQ: agx.CPUController.get_max_freq(),
                                S_GPU_FREQ: agx.GPUController.get_max_freq(),
                                S_MEMORY_FREQ: agx.MemoryController.get_max_freq()}

        elif self.control_scenario == 6:
            # Default control scenario that we employ in our experiments.
            agx.BoardController.reset_dvfs_settings()
        else:
            raise Exception(f"Unknown control scenario {self.control_scenario}")

    def pre_warmup(self, target_dvfs_config):
        pass

    def post_warmup(self):
        pass

    def pre_characterization(self, target_dvfs_config=None):
        if self.control_scenario in range(1, 6):
            return

        if target_dvfs_config is None:
            target_dvfs_config = self.dvfs_config
        else:
            self.dvfs_config = target_dvfs_config

        DVFSHelpers.set_dvfs_config(target_dvfs_config)
        if not DVFSHelpers.verify_dvfs_config(target_dvfs_config):
            logger.warning(f"DVFS config was not set correctly.")
            raise Exception(f"Failed to set DVFS config.")
        else:
            logger.info(f"Set DVFS config to {target_dvfs_config}")

    def post_characterization(self):
        if self.control_scenario in range(1, 6):
            return
        agx.BoardController.reset_dvfs_settings()

    def exit(self):
        pass


class TorchSettings:
    @staticmethod
    def setup(args: BenchmarkGenericArgs = None):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.set_num_threads(args.num_threads)
        if args is not None:
            if args.control_scenario == 5:
                torch._set_deterministic(True)  # Experimental flag affects a subset of pytorch operations
                # in version 1.6
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_optimization_config(fuser=JIT_FUSER):
        if fuser == 'fuser0':  # legacy fuser
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(False)
        elif fuser == 'fuser1':  # NNC
            old_profiling_executor = torch._C._jit_set_profiling_executor(True)
            old_profiling_mode = torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(True)
            torch._C._jit_set_texpr_fuser_enabled(True)
            torch._C._jit_set_nvfuser_enabled(False)
        elif fuser == 'fuser2':  # nvFuser
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_texpr_fuser_enabled(False)
            torch._C._jit_set_nvfuser_enabled(True)
        else:
            raise Exception("unrecognized fuser option")

        # Turn off auto-grad (needed for training only)
        torch.set_grad_enabled(False)

        # Optimized execution is on by default, but we want to make sure this is the case
        torch._C._set_graph_executor_optimize(True)

        # Verify settings
        optimized_execution_on = torch._C._get_graph_executor_optimize()
        autograd_off = torch.is_grad_enabled()
        if not optimized_execution_on or autograd_off:
            raise Exception(f"Couldn't apply Pytorch JIT settings. {optimized_execution_on} - {autograd_off}")


if __name__ == '__main__':
    print(Measurements.collected_enough_measurements(
        arr=[0.002245003237398129, 0.001498433899025187, 0.0019030931329882493, 0.0018162033068628964,
             0.0014470046040289564, 0.0013122918939745775, 0.0013108116796039992, 0.0015108169095912275,
             0.0015639764866533807, 0.0013181082200538065, 0.0013111661233808785, 0.001680114603197924,
             0.0016029603318205097, 0.0013410397383599793, 0.0013174827401723458, 0.0013163361564909597,
             0.0013119790763730724, 0.0013500664055541595, 0.0013313875912843388, 0.0013568416868824911,
             0.0014430853753602466, 0.0013109576818609082, 0.0013231948454915895, 0.0013428326150105132,
             0.0014181941650589437, 0.0013178996232122862, 0.0013096026566595518, 0.001314730908272709,
             0.0013314501858689499, 0.00192765086792191, 0.00134460452325181, 0.0013523803860046187,
             0.001315689863521812, 0.0016564325711626183, 0.0013718097140035723, 0.0013224859579378308,
             0.0013217980387933092, 0.0015449016024313067, 0.0015130266692817017, 0.001337620800403508,
             0.0013405186345600538, 0.0013164403772509448, 0.0013573837590916537, 0.0013305746382921448,
             0.0013200051621427753], return_rme=True))
