import glob
import numbers
import os
import pathlib
from os.path import exists
from os.path import join as jp

import numpy as np
import pandas as pd

from pyutils.characterization.common.types import KernelParameters
from pyutils.characterization.networks.properties import Parameters
from pyutils.characterization.kernels.utils.checks import KernelsChecks, is_kernel
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.characterization.networks.utils import get_unified_benchmark_name, is_network
from pyutils.common.config import FRONT_IGNORED_POWER_RECORDS, UPPER_LATENCY_THRESHOLD, IGNORED_TIMING_RECORDS, \
    ALL_LABELS
from pyutils.common.experiments_utils import Measurements
from pyutils.common.methods import get_metric, is_power, is_runtime, extract_metric_key, is_energy
from pyutils.common.paths import CHARACTERIZATION_DATA_DIR, MODELING_DATA_DIR
from pyutils.common.strings import S_NETWORK, S_PWR_SYNC_ONCE, \
    S_RUNTIME_MS, S_TIME_PER_RUN_MS, S_RUNTIME, S_POWER, S_ENERGY_J, S_BENCHMARK, S_PWR_READINGS, \
    S_DVFS_CONFIG, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_KERNEL_PARAMETERS, S_INTERNAL_PWR_METER, \
    S_EXTERNAL_PWR_METER, S_AVG_PWR_W, S_CONTROL_SCENARIO, S_RECORD_DATA, \
    S_PWR_METER, S_PARAMETERS, S_CHARACTERIZATION, S_MODELING, S_SUBMISSION_TIMESTAMP, S_RME, \
    S_KERNEL
from pyutils.common.strings import S_PARAMETERS_GENERATOR_VERSION
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger
from pyutils.hosts.agx import DVFSHelpers

logger = GlobalLogger().get_logger()

# Power is usually stable compared to runtime, fewer observations are enough for analysis
FILTERED_OBSERVATIONS_RATIO = {S_POWER: 3, S_RUNTIME: 5}
DISCARDED_WARMUP_OBSERVATIONS = {S_POWER: FRONT_IGNORED_POWER_RECORDS, S_RUNTIME: IGNORED_TIMING_RECORDS}


def check_dir(dir_path, ignore=False):
    if os.path.exists(dir_path):
        return dir_path
    else:
        if ignore:
            return dir_path
        else:
            raise FileNotFoundError(f'Directory: {dir_path} not found.')


def is_empty(dir_path):
    return len(os.listdir(dir_path)) == 0


def is_summarized(dir_path):
    for i in os.listdir(dir_path):
        if i.__contains__('summary'):
            return True, jp(dir_path, i)
    return False, None


class Helpers:
    @staticmethod
    def get_data_files(data_dir):
        data_files = []
        for file_name in os.listdir(data_dir):
            file_name = str(file_name)
            if file_name.__contains__('.json') and file_name.__contains__(f'_data') and \
                    not file_name.__contains__('summary') and not file_name.__contains__('aggregate'):
                data_files.append(file_name)
        return data_files


class DataAttributes:
    def __init__(self, node, benchmark=None, metric=S_RUNTIME, category=S_CHARACTERIZATION,
                 return_as='json', aggregate=False, overwrite_summary=False, data_dir: str = None,
                 return_placeholder=False, return_all_columns=False, sub_characterization_type=None,
                 filter_by_timestamp=False, overwrite_aggregate=False):
        self.node = node

        self.metric = get_metric(metric)
        self.return_as = return_as
        self.overwrite_summary = overwrite_summary
        self.overwrite_aggregate = overwrite_aggregate
        self.category = category
        self.aggregate = aggregate
        self.power_measurement_method = S_PWR_SYNC_ONCE
        self.return_placeholder = return_placeholder
        self.return_all_columns = return_all_columns
        if is_network(benchmark):
            self.benchmark = get_unified_benchmark_name(benchmark)
        elif is_kernel(benchmark):
            self.benchmark = KernelsChecks.get_unified_kernel_name(benchmark)
        else:
            self.benchmark = benchmark
        self.sub_characterization_type = sub_characterization_type
        self.filter_by_timestamp = filter_by_timestamp
        self.selected_timestamps = list()
        self.keep_submission_timestamp = False
        self.aggregate_file_ext = 'pickle'
        self.calculate_median_for_csv = True
        self.filter_by_power_meter = False
        self.power_meter = S_INTERNAL_PWR_METER
        self.check_sample_size = True

        if data_dir is None:
            self.data_dir = self.generate_data_dir()

    def generate_data_dir(self):
        if self.category == S_MODELING:
            p = check_dir(jp(MODELING_DATA_DIR, self.node, self.benchmark, self.metric))
        elif self.category == S_CHARACTERIZATION:
            if self.sub_characterization_type is None:
                p = check_dir(jp(CHARACTERIZATION_DATA_DIR, self.node, self.benchmark, self.metric))
            else:
                p = check_dir(jp(CHARACTERIZATION_DATA_DIR, self.node, "networks", 'kernels', self.benchmark,
                                 self.metric))
        else:
            raise Exception(f"Unknown data category: {self.category}")
        # Prepare directories
        if is_empty(p):
            raise Exception(f"Empty kernel data directory: {p}.")
        return p

    def set_metric(self, metric):
        self.metric = metric
        self.data_dir = self.generate_data_dir()

    def __str__(self):
        data = f"{self.node}:\n"
        data += f" {self.benchmark}:\n"
        data += f"  Metric:     {self.metric}\n"
        data += f"  Category:   {self.category}\n"
        data += f"  Aggregate:  {self.aggregate}\n"
        data += f"  Overwrite:  {self.overwrite_summary}\n"
        data += f"  Return:     {self.return_as}\n"
        data += f"  Directory:  {self.data_dir}\n"
        data += f"  Sub- Type:   {self.sub_characterization_type}\n"
        return data


class DataHandler:
    @staticmethod
    def get_power_key():
        return S_AVG_PWR_W

    @staticmethod
    def get_runtime_key():
        return S_RUNTIME_MS

    @staticmethod
    def get_energy_key():
        return S_ENERGY_J

    @staticmethod
    def get_metric_label(metric):
        if is_runtime(metric):
            return DataHandler.get_runtime_key()
        elif is_power(metric):
            return DataHandler.get_power_key()
        elif is_energy(metric):
            return DataHandler.get_energy_key()
        else:
            logger.warning(f"Could not map metric key to label: {metric}. Returning as is.")
            return metric

    @staticmethod
    def get_metric_key(metric):
        return DataHandler.get_metric_label(metric)

    @staticmethod
    def get_all_metrics_keys():
        return [DataHandler.get_runtime_key(), DataHandler.get_power_key(), DataHandler.get_energy_key()]

    @staticmethod
    def _summarize_runtime_single_record(attr: DataAttributes, record: dict) -> dict:
        ignore_latency_threshold = attr.category == S_CHARACTERIZATION
        # Special handling when the number of reported iterations is the set config
        # and not the actual iterations. We sum all readings after the first one, and if sum is
        # zero, then this is a dropped reading (or maybe we don't, worst case scenario we don't
        # have a mean for this particular config)
        if S_TIME_PER_RUN_MS in record.keys():
            runtime_key = S_TIME_PER_RUN_MS
        else:
            runtime_key = extract_metric_key(S_RUNTIME, record.keys())
        num_non_zeros = np.count_nonzero(record[runtime_key])
        # For very fast kernels (e.g., Size), running at maximum frequencies sometimes yield a strange
        # timing result where every other timing measurement is recorded as zero,
        # so this condition is modified to account for this behavior the relation used to be !=
        runtime_observations = np.array(record[runtime_key])
        if num_non_zeros < 0.5 * len(runtime_observations) or \
                (record[runtime_key][0] > UPPER_LATENCY_THRESHOLD and not ignore_latency_threshold):
            # Ignore latency threshold flag is there for very rare occasions
            # (e.g., DenseNet at lowest frequencies)
            return {}

        # Remove ignored samples
        runtime_observations = runtime_observations[IGNORED_TIMING_RECORDS:]
        non_zero_runtime_observations = runtime_observations[np.nonzero(runtime_observations)]
        return {DataHandler.get_runtime_key(): non_zero_runtime_observations.tolist()}

    @staticmethod
    def extract_dvfs_config(record: dict) -> dict:
        if S_DVFS_CONFIG in record.keys():
            dvfs_config = DVFSHelpers.extract_dvfs_config(record[S_DVFS_CONFIG])
        else:
            dvfs_config = DVFSHelpers.extract_dvfs_config(record)
        return dvfs_config

    @staticmethod
    def _summarize_power_single_record(record: dict) -> dict:
        if S_PWR_READINGS in record.keys():
            power_readings = record[S_PWR_READINGS]
        else:
            power_readings = record

        # Calculate average power
        if S_PWR_METER in record.keys() and record[S_PWR_METER] == S_EXTERNAL_PWR_METER:
            power_readings = power_readings['p_wattsup']
            power = np.array([float(i.replace('\n', '')) for i in power_readings])
        else:
            power = np.array(power_readings['p_cpu']) + np.array(power_readings['p_gpu']) + \
                    np.array(power_readings['p_soc']) + np.array(power_readings['p_ddr'])

        power = power[FRONT_IGNORED_POWER_RECORDS:]
        meter = record[S_PWR_METER]
        return {DataHandler.get_power_key(): power, S_PWR_METER: meter}

    @staticmethod
    def summarize_a_single_record(attr: DataAttributes, record: dict):
        metric = attr.metric
        record_summary = dict()
        record_summary = DataHandler.kernel_specific_processing(attr, record, record_summary)
        record_summary = DataHandler.network_specific_processing(attr, record, record_summary)

        dvfs_config = DataHandler.extract_dvfs_config(record)
        record_summary.update(dvfs_config)

        record_summary = DataHandler.add_timestamp(attr, record, record_summary)

        if is_power(metric):
            tmp = DataHandler._summarize_power_single_record(record)

        elif is_runtime(metric):
            tmp = DataHandler._summarize_runtime_single_record(attr, record)
        else:
            raise Exception(f"Unknown metric: {metric}. Aborting")

        record_summary.update(tmp)
        return record_summary

    @staticmethod
    def as_df(attr: DataAttributes):
        _, csv_file, _ = DataHandler.create_data_summary(attr)
        return pd.read_csv(csv_file).replace({np.nan: None})

    @staticmethod
    def as_full_df(attr: DataAttributes):
        _, _, pickle_file = DataHandler.create_data_summary(attr)
        return pd.read_pickle(pickle_file).replace({np.nan: None})

    @staticmethod
    def as_json(attr: DataAttributes):
        if not attr.aggregate:
            summary_json_file, _, _ = DataHandler.create_data_summary(attr)
        else:

            aggregate_file = DataHandler.create_data_aggregate(attr)
            return FileUtils.deserialize(aggregate_file, return_type='list')

        return FileUtils.deserialize(summary_json_file, return_type='list')

    @staticmethod
    def as_csv(attr: DataAttributes):
        return DataHandler.as_df(attr).to_csv()

    @staticmethod
    def is_outdated(data_dir: str, aggregate_file: str):
        if not exists(aggregate_file):
            return True

        # Last touch (creation/modification) of the aggregate file.
        latest_aggregate_file_touch_date = pathlib.Path(aggregate_file).stat().st_ctime
        if pathlib.Path(aggregate_file).stat().st_mtime >= latest_aggregate_file_touch_date:
            latest_aggregate_file_touch_date = pathlib.Path(aggregate_file).stat().st_mtime

        # Last touch (creation/modification) of all data files.
        latest_data_file_touch_date = -1
        for f in pathlib.Path(data_dir).iterdir():
            if str(f).__contains__("summary") or str(f).__contains__("aggregate"):
                continue
            create_time = pathlib.Path(f).stat().st_ctime
            mod_time = pathlib.Path(f).stat().st_mtime
            latest_touch_date = create_time if create_time > mod_time else mod_time
            latest_data_file_touch_date = latest_data_file_touch_date if \
                latest_data_file_touch_date >= latest_touch_date else latest_touch_date

        return latest_aggregate_file_touch_date < latest_data_file_touch_date

    @staticmethod
    def create_data_aggregate(attr: DataAttributes):
        """ Aggregate individual JSON files and save their content as a Pickle ser. object for faster
        manipulation. """
        output_aggregate_file = jp(attr.data_dir, f'{attr.category}_data_aggregate.{attr.aggregate_file_ext}')
        if os.path.exists(output_aggregate_file) and not attr.overwrite_aggregate and not \
                DataHandler.is_outdated(attr.data_dir, output_aggregate_file):
            pass
        else:
            logger.info("Re-creating the aggregate data file.")
            # Create a list of json data files and iterate through them all
            data_files = Helpers.get_data_files(attr.data_dir)
            records_aggregate = list()
            for file_name in data_files:
                data_file_path = jp(attr.data_dir, file_name)
                records = FileUtils.deserialize(data_file_path, return_type='list')
                for record in records:
                    if S_DVFS_CONFIG in record.keys():
                        dvfs_config = DVFSHelpers.extract_dvfs_config(record[S_DVFS_CONFIG])
                        record[S_DVFS_CONFIG].update(dvfs_config)
                    else:
                        dvfs_config = DVFSHelpers.extract_dvfs_config(record)
                        record.update(dvfs_config)

                    # Replace all Network keys with Benchmark to unify the code base terminology.
                    if S_NETWORK in record.keys():
                        record.update({S_BENCHMARK: record[S_NETWORK]})
                        record.pop(S_NETWORK)

                    records_aggregate.append(record)

            FileUtils.serialize(records_aggregate, output_aggregate_file, save_as='list', append=False)
        return output_aggregate_file

    @staticmethod
    def ignore_record(attr: DataAttributes, record: dict):
        if attr.filter_by_timestamp:
            if S_SUBMISSION_TIMESTAMP in record.keys() and \
                    record[S_SUBMISSION_TIMESTAMP] not in attr.selected_timestamps:
                return True

            if S_SUBMISSION_TIMESTAMP not in record.keys():
                return True

        if is_power(attr.metric) and attr.filter_by_power_meter:
            # Default records do not have this key because we always use the internal sensors
            if S_PWR_METER not in record.keys():
                if attr.power_meter != S_INTERNAL_PWR_METER:
                    return True

            elif record[S_PWR_METER] != attr.power_meter:
                return True
        return False

    @staticmethod
    def convert_non_primitive_to_str(d):
        if isinstance(d, numbers.Number) or isinstance(d, str):
            return d
        else:
            return str(d)

    @staticmethod
    def add_timestamp(attr: DataAttributes, record: dict, record_summary: dict):
        if attr.keep_submission_timestamp:
            if S_SUBMISSION_TIMESTAMP in record.keys():
                record_summary[S_SUBMISSION_TIMESTAMP] = record[S_SUBMISSION_TIMESTAMP]
            else:
                record_summary[S_SUBMISSION_TIMESTAMP] = '0101011900'
        return record_summary

    @staticmethod
    def is_complete_record(record: dict) -> bool:
        if S_BENCHMARK in record.keys() and S_SUBMISSION_TIMESTAMP in record.keys():
            return True
        return False

    @staticmethod
    def parse_kernel_parameters(attr: DataAttributes, record: dict) -> KernelParameters:
        if DataHandler.is_complete_record(record):
            if S_KERNEL_PARAMETERS in record.keys():
                parameters_obj = \
                    KernelsLoaders.load_kernel_parameters_parser(attr.benchmark)(record[S_KERNEL_PARAMETERS])
            elif S_PARAMETERS in record.keys():
                parameters_obj = KernelsLoaders.load_kernel_parameters_parser(attr.benchmark)(record[S_PARAMETERS])
            else:
                raise Exception(f"Unhandled situation here: {record}")
        else:
            # TODO - Assumption: we assume that the parameters are spread in the dict record.
            parameters_obj = KernelsLoaders.load_kernel_parameters_parser(attr.benchmark)(record)
        return parameters_obj

    @staticmethod
    def kernel_specific_processing(attr: DataAttributes, record: dict, record_summary: dict = None):
        if record_summary is None:
            record_summary = dict()

        if not is_kernel(attr.benchmark):
            return record_summary
        parameters_obj = DataHandler.parse_kernel_parameters(attr, record)

        # We use S_PARAMETERS instead of kernel parameters to unify the naming scheme for kernels,
        # and benchmarks.
        record_summary[S_BENCHMARK] = KernelsChecks.get_unified_kernel_name(attr.benchmark)
        record_summary[S_PARAMETERS] = parameters_obj.to_str()
        return record_summary

    @staticmethod
    def network_specific_processing(attr: DataAttributes, record: dict, record_summary: dict = None):
        if record_summary is None:
            record_summary = dict()

        if not is_network(attr.benchmark):
            return record_summary

        if S_PARAMETERS in record.keys():
            parameters_obj = Parameters.from_dict(record[S_PARAMETERS])
        else:
            # TODO - Assumption: we assume that the parameters are spread in the dict record.
            parameters_obj = Parameters.from_dict(record)

        # We use S_PARAMETERS instead of kernel parameters to unify the naming scheme for kernels, and
        # benchmarks.
        record_summary[S_BENCHMARK] = get_unified_benchmark_name(attr.benchmark)
        record_summary[S_PARAMETERS] = parameters_obj.to_str()
        return record_summary

    @staticmethod
    def create_data_summary(attr: DataAttributes):

        if attr.keep_submission_timestamp:
            output_json_file_name = f'{attr.category}_data_summary_with_timestamp.json'
            output_csv_file_name = f'{attr.category}_data_summary_with_timestamp.csv'
            output_pickle_file_name = f'{attr.category}_data_summary_with_timestamp.pickle'
        else:
            output_json_file_name = f'{attr.category}_data_summary.json'
            output_csv_file_name = f'{attr.category}_data_summary.csv'
            output_pickle_file_name = f'{attr.category}_data_summary.pickle'

        output_json_file = jp(attr.data_dir, output_json_file_name)
        output_csv_file = jp(attr.data_dir, output_csv_file_name)
        output_pickle_file = jp(attr.data_dir, output_pickle_file_name)

        aggregate_file = jp(attr.data_dir, f'{attr.category}_data_aggregate.{attr.aggregate_file_ext}')

        attr.overwrite_aggregate = DataHandler.is_outdated(attr.data_dir, aggregate_file)
        if exists(output_json_file) and exists(output_csv_file) and exists(output_pickle_file) \
                and not attr.overwrite_summary and not attr.overwrite_aggregate:
            pass
        else:
            # Sanity check: delete the summary files if they exist
            FileUtils.silent_remove(output_json_file)
            FileUtils.silent_remove(output_csv_file)

            # Load aggregated data file content
            if not exists(aggregate_file) or attr.overwrite_aggregate:
                DataHandler.create_data_aggregate(attr)

            records = FileUtils.deserialize(aggregate_file, return_type='list')
            records_summary = list()
            for record in records:
                if DataHandler.ignore_record(attr, record):
                    continue
                # if record[S_SUBMISSION_TIMESTAMP] != "572125082022" and attr.benchmark == "mobilenet_v2":
                #     continue
                record_summary = DataHandler.summarize_a_single_record(attr, record)
                if record_summary:
                    records_summary.append(record_summary)

            df = pd.DataFrame(records_summary)

            # The key columns that we use to detect duplicates
            key_columns = [col for col in df.columns if col not in ALL_LABELS]

            duplicated_rows = df[df.duplicated(subset=key_columns, keep=False)].copy()

            # Sometimes we have duplicates, we keep the samples with the lowest RME.
            if duplicated_rows.shape[0] != 0:
                duplicated_rows[S_RME] = [Measurements.median_relative_margin_of_error(i, confidence_level_percent=99)
                                          for i in duplicated_rows[DataHandler.get_metric_label(attr.metric)]]
                duplicated_rows.sort_values(by=[S_RME], ascending=True, inplace=True)
                duplicated_rows.drop_duplicates(inplace=True, keep='first', subset=key_columns, ignore_index=True)
                duplicated_rows.drop(columns=[S_RME], inplace=True)

            # Drop the duplicate rows from the original data frame
            df.drop_duplicates(subset=key_columns, inplace=True, ignore_index=True, keep=False)

            # Insert the filtered duplicated rows after we select the samples with the lowest RME
            df = pd.concat([df, duplicated_rows], ignore_index=True)

            df.to_pickle(output_pickle_file)
            df.to_json(output_json_file, orient='records', lines=True)

            # We save the median of the metric in the CSV file
            existing_labels = list()
            for col in df.columns:
                if col in ALL_LABELS:
                    existing_labels.append(col)

            for i in range(df.shape[0]):
                for label in existing_labels:
                    df.at[i, label] = np.median(df.at[i, label])
            df.sort_values(by=[S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ], inplace=True)
            df.to_csv(output_csv_file, index=False)
        return output_json_file, output_csv_file, output_pickle_file

    @staticmethod
    def get_benchmark_data(attr: DataAttributes):
        if attr.return_as == 'json':
            return DataHandler.as_json(attr)
        elif attr.return_as == 'df':
            return DataHandler.as_df(attr)
        elif attr.return_as == 'full_df':
            return DataHandler.as_full_df(attr)

    @staticmethod
    def get_nw_kernel_data(attr: DataAttributes):
        attr.sub_characterization_type = S_KERNEL
        return DataHandler.get_benchmark_data(attr)

    @staticmethod
    def get_kernel_data(attr: DataAttributes):
        attr.sub_characterization_type = None
        return DataHandler.get_benchmark_data(attr)

    @staticmethod
    def get_kernel_runtime(attr: DataAttributes, dvfs_configs, kernel_parameters: dict):
        """ This method is used to extract the kernel runtime from the characterization data, if it exists,
        for a specific DVFS config and kernel parameters. A caveat here is that it targets only the kernels that are
        used (same parameters) within a specific network and not a generic method (for any kernel parameters). """
        attr.set_metric(S_RUNTIME)
        attr.sub_characterization_type = S_KERNEL
        return DataHandler._get_kernel_metric_data(attr, dvfs_configs, kernel_parameters)

    @staticmethod
    def get_kernel_power(attr: DataAttributes, dvfs_configs, kernel_parameters: dict):
        """ This method is used to extract the kernel power from the characterization data, if it exists,
        for a specific DVFS config and kernel parameters. A caveat here is that it targets only the kernels that are
        used (same parameters) within a specific network and not a generic method (for any kernel parameters). """
        attr.set_metric(S_POWER)
        attr.sub_characterization_type = S_KERNEL
        return DataHandler._get_kernel_metric_data(attr, dvfs_configs, kernel_parameters)

    @staticmethod
    def get_kernel_energy(attr: DataAttributes, dvfs_configs, kernel_parameters: dict):
        """ This method is used to extract the kernel energy from the characterization data, if it exists,
        for a specific DVFS config and kernel parameters. A caveat here is that it targets only the kernels that are
        used (same parameters) within a specific network and not a generic method (for any kernel parameters). """
        avg_power = DataHandler.get_kernel_power(attr, dvfs_configs, kernel_parameters)
        runtime_ms = DataHandler.get_kernel_runtime(attr, dvfs_configs, kernel_parameters)
        combined = avg_power.merge(runtime_ms, how='left', on=[S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ])
        combined[S_ENERGY_J] = combined[S_RUNTIME_MS] * combined[S_AVG_PWR_W] / 1e3
        combined.drop(inplace=True, columns=[S_RUNTIME_MS, S_AVG_PWR_W])
        return combined

    @staticmethod
    def _get_kernel_metric_data(attr: DataAttributes, dvfs_configs, kernel_parameters: dict):
        """ This method is used to extract the kernel metric data from the characterization data, if it exists,
        for a specific DVFS config and kernel parameters. A caveat here is that it targets only the kernels that are
        used (same parameters) within a specific network and not a generic method (for any kernel parameters). """
        attr.aggregate = False
        attr.return_as = 'df'

        if type(dvfs_configs) is not list:
            dvfs_configs = [dvfs_configs]

        kernel_details = DataHandler.kernel_specific_processing(attr, kernel_parameters)

        tmp = list()
        for c in dvfs_configs:
            tmp.append({**kernel_details, **c})
        search_criteria = pd.DataFrame(tmp)
        search_criteria.drop(inplace=True, columns=[S_PARAMETERS_GENERATOR_VERSION], errors="ignore")

        records = DataHandler.get_nw_kernel_data(attr)
        records.drop(inplace=True, columns=[S_PARAMETERS_GENERATOR_VERSION], errors="ignore")
        selected_records = search_criteria.merge(records, how='left', on=list(search_criteria.columns))

        metric_key = extract_metric_key(attr.metric, records.columns)

        if selected_records.shape[0] != 0:
            if attr.return_all_columns:
                return selected_records
            return selected_records[metric_key]

        if attr.return_placeholder:
            return None

        raise Exception(
            f"{attr.metric} for '{attr.benchmark}' -> {kernel_parameters}:{dvfs_configs} not found.")

    @staticmethod
    def _get_nw_metric_data(attr: DataAttributes, target_dvfs_configs):
        # The assumption is that we use the summary json form for networks, so we make sure of that here
        attr.aggregate = False
        if type(target_dvfs_configs) is not list:
            target_dvfs_configs = [target_dvfs_configs]

        target_dvfs_configs = pd.DataFrame(target_dvfs_configs)
        attr.return_as = 'df'
        records = DataHandler.get_benchmark_data(attr)
        print(records)
        selected_record = target_dvfs_configs.merge(records, how='left', on=list(target_dvfs_configs.columns))
        metric_key = extract_metric_key(attr.metric, records.columns)
        if selected_record.shape[0] != 0:
            if attr.return_all_columns:
                return selected_record
            return selected_record[metric_key]

        if attr.return_placeholder:
            return None
        raise Exception(
            f"Could not find data for {attr.metric} of {attr.benchmark} on {attr.node} at {target_dvfs_configs}")

    @staticmethod
    def get_nw_runtime(attr: DataAttributes, dvfs_configs):
        attr.set_metric(S_RUNTIME)
        return DataHandler._get_nw_metric_data(attr, dvfs_configs)

    @staticmethod
    def get_nw_power(attr: DataAttributes, dvfs_configs):
        attr.set_metric(S_POWER)
        return DataHandler._get_nw_metric_data(attr, dvfs_configs)

    @staticmethod
    def get_nw_energy(attr: DataAttributes, target_dvfs_config):
        # The assumption is that we use the summary json form for networks, so we make sure of that here
        runtime_ms = DataHandler.get_nw_runtime(attr, target_dvfs_config)
        avg_pwr = DataHandler.get_nw_power(attr, target_dvfs_config)
        if avg_pwr is None:
            return None
        combined = avg_pwr.merge(runtime_ms, how='left', on=[S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ])
        combined[S_ENERGY_J] = combined[S_RUNTIME_MS] * combined[S_AVG_PWR_W] / 1e3
        combined.drop(inplace=True, columns=[S_RUNTIME_MS, S_AVG_PWR_W])
        return combined

    @staticmethod
    def clear_cached_summary(category: None):
        if category is None:
            logger.warning("None was specified for clearing the cached summary. Ignoring and existing.")
            return

        if category not in [S_CHARACTERIZATION, S_MODELING]:
            logger.error(f"Invalid category: {category} for clearing")
            return

        # Remove the summary and aggregate files of the specified category
        data_parent_dir = CHARACTERIZATION_DATA_DIR if category is S_CHARACTERIZATION else MODELING_DATA_DIR

        for node_dir in os.listdir(data_parent_dir):
            node_dir_path = jp(data_parent_dir, node_dir)
            for sub_dir in os.listdir(node_dir_path):
                sub_dir_path = jp(node_dir_path, sub_dir)
                summary_files = glob.glob(f'{sub_dir_path}/**/*_summary*', recursive=True)
                aggregate_files = glob.glob(f'{sub_dir_path}/**/*_aggregate*', recursive=True)

                for summary_file in summary_files:
                    os.remove(summary_file)

                for aggregate_file in aggregate_files:
                    os.remove(aggregate_file)


class Tests:
    @staticmethod
    def test_get_nw_kernel_data():
        metric = 'runtime'
        data_attr = DataAttributes(node='node-15', benchmark='hardtanh', metric=metric,
                                   category=S_CHARACTERIZATION, return_as='full_df', overwrite_summary=True)
        d = DataHandler.get_nw_kernel_data(data_attr)
        print(d)

    @staticmethod
    def test_summarize_all_nw_kernels_data():
        metric = 'runtime'
        node = "node-15"
        for f in os.listdir(jp(CHARACTERIZATION_DATA_DIR, node, "networks", "kernels")):
            logger.info(f"Summarizing: {f}")
            data_attr = DataAttributes(node='node-15', benchmark=f, metric=metric,
                                       category=S_CHARACTERIZATION, return_as='full_df', overwrite_summary=True)
            d = DataHandler.get_nw_kernel_data(data_attr)
            print(d)

    @staticmethod
    def test_get_nw_data():
        metric = 'power'
        attr = DataAttributes(node='node-01', aggregate=False, overwrite_summary=True, category=S_CHARACTERIZATION,
                              metric=metric, benchmark='shufflenetv2', return_as='df')
        data = DataHandler.get_benchmark_data(attr)
        print(data.shape)

    @staticmethod
    def test_get_nw_runtime():
        dvfs_config = {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000}
        dvfs_config = {'cpu_freq': 115200, 'gpu_freq': 114750000, 'memory_freq': 204000000}
        attr = DataAttributes(node='node-02', overwrite_summary=True, category=S_CHARACTERIZATION, benchmark='densenet')
        runtime = DataHandler.get_nw_runtime(attr, dvfs_config)
        logger.info(f"Network runtime: {runtime}")

    @staticmethod
    def test_get_nw_power():
        dvfs_config = {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000}
        attr = DataAttributes(node='node-03', overwrite_summary=True, category=S_CHARACTERIZATION,
                              benchmark='squeezenet')
        power = DataHandler.get_nw_power(attr, dvfs_config)
        logger.info(f"Network power: {power}")

    @staticmethod
    def test_get_nw_energy():
        dvfs_config = {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000}
        attr = DataAttributes(node='node-02', overwrite_summary=True, category=S_CHARACTERIZATION, benchmark='resnet')
        power = DataHandler.get_nw_energy(attr, dvfs_config)
        logger.info(f"Network energy: {power}")

    @staticmethod
    def test_get_kernel_runtime():
        dvfs_config = [{"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000},
                       {"cpu_freq": 1190400, "gpu_freq": 675750000, "memory_freq": 1065600000}]
        kernels_parameters = {S_KERNEL_PARAMETERS:
                                  {"input_tensors": [{"count": 2, "tensor_shape": [1, 64, 55, 55]}], "dim": 1,
                                   "generator_version": -1}}

        attr = DataAttributes(node='node-15', benchmark='cat',
                              category=S_CHARACTERIZATION, return_as='df', overwrite_summary=False)
        attr.return_all_columns = True
        r = DataHandler.get_kernel_runtime(attr, dvfs_config, kernel_parameters=kernels_parameters)
        logger.info(f"Kernel runtime is: {r}")

    @staticmethod
    def test_get_kernel_power():
        dvfs_config = {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000}
        kernels_parameters = {"n": 1, "c": 3, "h": 256, "w": 256, "in_channels": 3, "out_channels": 64, "kernel_h": 11,
                              "kernel_w": 11, "bias": 1, "stride_h": 4, "stride_w": 4, "padding_h": 2, "padding_w": 2,
                              "dilation_h": 1, "dilation_w": 1, "groups": 1, "padding_mode": "zeros",
                              "generator_version": -1}
        attr = DataAttributes(node='node-02', benchmark='conv2d',
                              category=S_CHARACTERIZATION, return_as='json', overwrite_summary=True)
        r = DataHandler.get_kernel_power(attr, dvfs_config, kernel_parameters=kernels_parameters)
        logger.info(f"Kernel Conv2d power is: {r}")

    @staticmethod
    def test_get_kernel_energy():
        dvfs_config = {"cpu_freq": 2265600, "gpu_freq": 1377000000, "memory_freq": 2133000000}
        kernels_parameters = {'n': 1, 'c': 64, 'h': 56, 'w': 56, 'in_channels': 64, 'out_channels': 64, 'kernel_h': 1,
                              'kernel_w': 1, 'bias': 1, 'stride_h': 1, 'stride_w': 1, 'padding_h': 0, 'padding_w': 0,
                              'dilation_h': 1, 'dilation_w': 1, 'groups': 1, 'padding_mode': 'zeros',
                              'generator_version': -1}
        attr = DataAttributes(node='node-02', benchmark='conv2d',
                              category=S_CHARACTERIZATION, return_as='json', overwrite_summary=True)
        r = DataHandler.get_kernel_energy(attr, dvfs_config, kernel_parameters=kernels_parameters)
        logger.info(f"Kernel Conv2d energy is: {r}")

    @staticmethod
    def test_get_kernel_data():
        attr = DataAttributes(node='node-13', benchmark='conv2d', category=S_MODELING, metric=S_RUNTIME,
                              overwrite_summary=True,
                              aggregate=True)
        r = DataHandler.get_kernel_data(attr)
        print("Done.")

    @staticmethod
    def test_clear_summary():
        DataHandler.clear_cached_summary(category=S_CHARACTERIZATION)

    @staticmethod
    def test_get_benchmark_data():
        metric = 'runtime'
        attr = DataAttributes(node='node-01', aggregate=False, overwrite_summary=True, category=S_CHARACTERIZATION,
                              metric=metric, return_as='df', benchmark='lud')
        data = DataHandler.get_benchmark_data(attr)
        print(data.shape)

    @staticmethod
    def test_hand_written():
        attr = DataAttributes(benchmark="bfs", node="node-01", category=S_CHARACTERIZATION,
                              overwrite_aggregate=True, overwrite_summary=False, aggregate=True,
                              metric=S_POWER, return_as='json', filter_by_timestamp=False)
        attr.check_sample_size = False
        bm_data = DataHandler.get_benchmark_data(attr)
        for record in bm_data:
            if S_RECORD_DATA in record.keys() and not record[S_RECORD_DATA]:
                continue

            if S_CONTROL_SCENARIO not in record.keys() or record[S_CONTROL_SCENARIO] != 6:
                continue

            print(record)

    @staticmethod
    def test_all():
        Tests.test_get_nw_kernel_data()
        Tests.test_get_nw_power()
        Tests.test_get_nw_energy()
        Tests.test_get_kernel_runtime()
        Tests.test_get_kernel_power()
        Tests.test_get_kernel_energy()
        Tests.test_get_kernel_data()


if __name__ == '__main__':
    # Tests.test_hand_written()
    # Tests.test_all()
    # Tests.test_get_benchmark_data()
    # Tests.test_get_nw_data()
    # Tests.test_get_nw_power()
    # Tests.test_get_nw_kernel_data()
    # print(d)
    Tests.test_clear_summary()
    # Tests.test_get_kernel_runtime()
    Tests.test_get_nw_kernel_data()
    # Tests.test_get_nw_runtime()
    # Tests.test_get_kernel_data()
    # Tests.test_summarize_all_nw_kernels_data()
