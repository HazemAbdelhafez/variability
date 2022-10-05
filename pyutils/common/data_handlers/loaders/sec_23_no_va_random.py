# Load data from current experiments for the future submission.
from typing import List, Dict, Union

import pandas as pd

from pyutils.characterization.networks.utils import is_network
from pyutils.hosts.agx import DVFSHelpers
from pyutils.common.strings import S_CHARACTERIZATION, S_RECORD_DATA, S_CONTROL_SCENARIO, S_EXP_NOTES, \
    S_NODE_ID, S_CPU_FREQ, S_BENCHMARK, \
    S_WARMUP_ITERATIONS, S_NUM_OBSERVATIONS, S_SUBMISSION_TIMESTAMP, S_CORE_AFFINITY, S_NUM_ISOLATED_CORES, \
    S_NUM_THREADS, S_RME, S_CONFIDENCE_LVL, S_BLOCK_RUNTIME_MS, S_TIMING_METHOD, S_BLOCK_SIZE, \
    S_RUNTIME, S_RUNTIME_MS
from pyutils.common.config import FRONT_IGNORED_POWER_RECORDS, TAIL_IGNORED_POWER_RECORDS, S_DVFS_COLS, \
    ALL_LABELS, ALL_NODES, MIN_CPU_FREQ, IGNORED_TIMING_RECORDS, CONFIDENCE_LVL
from pyutils.common.data_handlers.data_interface import DataAttributes, DataHandler
from pyutils.common.experiments_utils import node_id_to_int, Measurements, get_missing_configs_from_df
from pyutils.common.methods import dict_to_columns, is_power, is_runtime

# Pandas display parameters
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def is_valid_record(record: dict):
    """ Check that a given record belong to the data snapshot of ISPASS 22/SEC 22 submission"""
    return record[S_EXP_NOTES] == "runtime_v1" and "environment_variables" in record.keys()


def remove_duplicates(df: pd.DataFrame):
    # The key columns that we use to detect duplicates, we know from the is_ispass_22_record method that the
    # rest are the same, so no need to check on them.
    key_columns = [S_BENCHMARK, S_NODE_ID] + S_DVFS_COLS
    duplicated_rows = df[df.duplicated(subset=key_columns, keep=False)].copy()

    if duplicated_rows.shape[0] != 0:
        duplicated_rows.sort_values(by=[S_RME], ascending=True, inplace=True)
        duplicated_rows.drop_duplicates(inplace=True, keep='first', subset=key_columns, ignore_index=True)

    # Drop the duplicate rows from the original data frame
    df.drop_duplicates(subset=key_columns, inplace=True, ignore_index=True, keep=False)

    # Insert the filtered duplicated rows after we select only the largest samples
    df = pd.concat([df, duplicated_rows])
    return df


def get_benchmark_data(benchmark, node, metric, exclude_lower_cpu_freqs=True) -> pd.DataFrame:
    """ Parse and combine data (as list of records) for one benchmark on one node. """

    attr = DataAttributes(benchmark=benchmark, node=node, category=S_CHARACTERIZATION,
                          aggregate=True, metric=metric, return_as='json')
    attr.keep_submission_timestamp = True
    return get_benchmark_data_from_attribute(attr, exclude_lower_cpu_freqs)


def get_benchmark_data_from_attribute(attr: DataAttributes, exclude_lower_cpu_freqs: bool = True) \
        -> pd.DataFrame:
    metric = DataHandler.get_metric_key(attr.metric)
    nw_data: list = DataHandler.get_benchmark_data(attr)
    records = list()
    for record in nw_data:

        if S_RECORD_DATA in record.keys() and not record[S_RECORD_DATA]:
            continue

        if S_CONTROL_SCENARIO not in record.keys() or record[S_CONTROL_SCENARIO] != 6:
            continue

        if is_runtime(metric):
            # Special handling for ispass_22 submission data
            if not is_valid_record(record):
                continue
            record[metric] = record[metric][IGNORED_TIMING_RECORDS:]
            record.pop("environment_variables")
            record.pop("memory_maps")
            record.pop("pid")

        record = dict_to_columns(record)
        if is_power(metric):
            if S_EXP_NOTES not in record.keys() or not str(record[S_EXP_NOTES]).__contains__("ispass_22"):
                continue
            record[metric] = record['p_all'][FRONT_IGNORED_POWER_RECORDS: -TAIL_IGNORED_POWER_RECORDS]

        if exclude_lower_cpu_freqs and record[S_CPU_FREQ] < MIN_CPU_FREQ:
            continue

        # Calculate RME for each sample
        record["required_rme"] = record.pop(S_RME)
        record[S_RME] = Measurements.median_relative_margin_of_error(record[metric], CONFIDENCE_LVL)

        record.update({S_NODE_ID: node_id_to_int(attr.node)})

        records.append(record)

    df = pd.DataFrame(records)

    df = remove_duplicates(df)
    df = df.sort_values(by=S_DVFS_COLS)

    return df


def get_benchmarks_data(benchmarks, nodes=ALL_NODES, metric=S_RUNTIME, return_as: str = "records",
                        check_size: bool = True) -> Union[List[Dict], pd.DataFrame]:
    """ Parse and combine data (as list of records) for multiple benchmarks on multiple nodes. """
    records = list()
    for node in nodes:
        for benchmark in benchmarks:
            bm_node_records = get_benchmark_data(benchmark, node, metric)
            bm_node_records = bm_node_records.reset_index(drop=True)
            l, t = get_missing_configs_from_df(bm_node_records, "dvfs_455.json")
            assert len(l) == 0
            records.append(bm_node_records)
    df = pd.concat(records)
    df = df.reset_index(drop=True)

    if check_size:
        # Get unique frequency configurations
        num_unique_frequency_configuration = DVFSHelpers.count_unique_dvfs_configs(df)
        assert df.shape[0] == int(num_unique_frequency_configuration * len(nodes) * len(benchmarks))

    if return_as == "records":
        return df.to_dict(orient='records')
    else:
        return df


if __name__ == '__main__':
    d = get_benchmarks_data(benchmarks=["mobilenetv2"], return_as='df')
