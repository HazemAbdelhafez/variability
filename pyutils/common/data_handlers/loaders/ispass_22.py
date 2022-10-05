# Load data from ISPASS 22 submission.
from typing import List, Dict, Union

import pandas as pd

from pyutils.characterization.networks.utils import is_network
from pyutils.hosts.agx import DVFSHelpers
from pyutils.common.strings import S_CHARACTERIZATION, S_RECORD_DATA, S_CONTROL_SCENARIO, S_EXP_NOTES, \
     S_NODE_ID, S_CPU_FREQ, S_BENCHMARK, \
    S_WARMUP_ITERATIONS, S_NUM_OBSERVATIONS, S_SUBMISSION_TIMESTAMP, S_CORE_AFFINITY, S_NUM_ISOLATED_CORES, \
    S_NUM_THREADS, S_RME, S_CONFIDENCE_LVL, S_BLOCK_RUNTIME_MS, S_TIMING_METHOD, S_BLOCK_SIZE, \
    S_RUNTIME
from pyutils.common.config import FRONT_IGNORED_POWER_RECORDS, TAIL_IGNORED_POWER_RECORDS, S_DVFS_COLS, \
    ALL_LABELS, ALL_NODES, MIN_CPU_FREQ, IGNORED_TIMING_RECORDS, CONFIDENCE_LVL
from pyutils.common.data_handlers.data_interface import DataAttributes, DataHandler
from pyutils.common.experiments_utils import node_id_to_int, Measurements
from pyutils.common.methods import dict_to_columns, is_power, is_runtime


def is_ispass_22_record(record: dict):
    """ Check that a given record belong to the data snapshot of ISPASS 22/SEC 22 submission"""
    if S_BENCHMARK not in record.keys():
        return False
    must_exist_keys = [S_BENCHMARK, S_WARMUP_ITERATIONS, S_NUM_OBSERVATIONS, S_SUBMISSION_TIMESTAMP, S_CONTROL_SCENARIO,
                       S_CORE_AFFINITY, S_NUM_ISOLATED_CORES, S_NUM_THREADS, S_RECORD_DATA, S_RME, S_CONFIDENCE_LVL,
                       S_BLOCK_RUNTIME_MS, S_TIMING_METHOD, S_BLOCK_SIZE]

    for i in must_exist_keys:
        if i not in record.keys():
            return False

    if is_network(record[S_BENCHMARK]):
        c = True
        c = c and record[S_WARMUP_ITERATIONS] == 5
        c = c and record[S_NUM_OBSERVATIONS] == 50
        c = c and record[S_CONTROL_SCENARIO] == 6
        c = c and record[S_CORE_AFFINITY]
        c = c and record[S_NUM_ISOLATED_CORES] == 6
        c = c and record[S_NUM_THREADS] == 6
        c = c and record[S_RECORD_DATA]
        c = c and record[S_RME] == 0.5
        c = c and record[S_CONFIDENCE_LVL] == 99
        c = c and record[S_BLOCK_RUNTIME_MS] == 1000
        c = c and record[S_TIMING_METHOD] == 'block_based'
        c = c and record[S_BLOCK_SIZE] == 100
        if "batch_size" in record.keys():
            c = c and record["batch_size"] == 1
        return c
    else:
        if S_EXP_NOTES not in record.keys():
            return False
        return record[S_EXP_NOTES] == "ispass_22_v1"


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

        record = dict_to_columns(record)
        if is_power(metric):
            if S_EXP_NOTES not in record.keys() or not str(record[S_EXP_NOTES]).__contains__("ispass_22"):
                continue
            record[metric] = record['p_all'][FRONT_IGNORED_POWER_RECORDS: -TAIL_IGNORED_POWER_RECORDS]

        if is_runtime(metric):
            # Special handling for ispass_22 submission data
            # TODO: this needs better handling for both runtime and power... for all experiments
            if not is_ispass_22_record(record):
                continue

        if exclude_lower_cpu_freqs and record[S_CPU_FREQ] < MIN_CPU_FREQ:
            continue

        record[metric] = record[metric][IGNORED_TIMING_RECORDS:]

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
    d = get_benchmarks_data(benchmarks=["vgg"], return_as='df')
    print(d)

