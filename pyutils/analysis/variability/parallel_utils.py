import os.path

import dask
import numpy as np
import pandas as pd

from pyutils.analysis.common.error_metrics import Metrics
from pyutils.analysis.variability.strings import S_MEDIAN_OF_ALL, S_SAMPLE_MEDIAN, S_INTRA, S_INTER
from pyutils.common.config import MIN_CPU_FREQ, S_DVFS_COLS
from pyutils.common.data_handlers.data_interface import DataAttributes
import pyutils.common.data_handlers.loaders.sec_23 as sec_23
import pyutils.common.data_handlers.loaders.sec_23_no_va_random as sec_23_no_va_random
from pyutils.common.experiments_utils import node_id_to_int
from pyutils.common.paths import CACHE_DIR
from pyutils.common.strings import S_CPU_FREQ, S_RME, S_METRIC, S_BENCHMARK, S_CHARACTERIZATION, \
    S_KEEP_SUBMISSION_TIMESTAMP, S_NODE_ID
from pyutils.common.utils import get_pymodule_dir_path
from pyutils.common.utils import prepare, FileUtils
from pyutils.run.analysis.config import S_EXCLUDE_LOWEST_CPU_FREQS, S_OVERWRITE_AGGREGATE, \
    S_OVERWRITE_SUMMARY, S_FILTER_BY_TS, \
    S_SELECTED_TS


# Methods for parallel implementation of the analysis.

@dask.delayed
def par_calculate_sample_median(records: pd.DataFrame, metric_key: str):
    """ A parallel method to calculate the median of each sample in a list of input samples. """
    records[S_SAMPLE_MEDIAN] = records[metric_key].apply(np.median)
    return records


@dask.delayed
def par_filter_out_lowest_cpu_freqs(records: pd.DataFrame, cfg):
    """ Remove the lowest CPU configs from the data if they exist. """
    if S_EXCLUDE_LOWEST_CPU_FREQS in cfg.keys() and cfg[S_EXCLUDE_LOWEST_CPU_FREQS]:
        return records.drop(index=records[records[S_CPU_FREQ] < MIN_CPU_FREQ].index)
    else:
        return records


def select_min_num_obs(x, min_num_obs):
    """ A method to unify the number of observations (pick minimum number) across all samples. """
    n = len(x)
    if n < min_num_obs:
        raise Exception(f"Un-matching min number of observations: input of length {n}, and required min: "
                        f"{min_num_obs}")
    else:
        return x[:min_num_obs]


@dask.delayed
def par_select_min_num_obs(df: pd.DataFrame, metric_key: str, min_num_obs: int):
    """ A parallel method that calls select_min_num_obs on the dask data frame partitions in parallel. """
    df[metric_key] = df[metric_key].apply(select_min_num_obs, args=(min_num_obs,))
    return df


@dask.delayed
def par_remove_metric_outliers(df: pd.DataFrame, metric_key: str):
    """ A parallel method that calls removes outliers on a per record on the dask data frame partitions in
    parallel. """

    def remove_outliers_v3(x, rme=0.5):
        m = np.median(x)
        return [i for i in x if 100 * abs(i - m) / m <= rme]

    df = df.drop(index=df[df[S_RME] > df["required_rme"]].index)
    df[metric_key] = df[metric_key].apply(lambda x: remove_outliers_v3(x))

    return df


@dask.delayed
def par_get_benchmark_data_for_node(node: str, cfg: dict, get_bm_data_method):
    # TODO: We used get_bm_data_method in the past to get Rodinia or Networks data. Now I replaced it
    # with a single method that only gets Networks data. Fix this for Rodinia.
    """ A dask method that extracts a benchmark data from its aggregate form for a specific node. """
    cached_file = prepare(get_pymodule_dir_path(CACHE_DIR, __file__), cfg[S_METRIC], node,
                          f'{cfg[S_BENCHMARK]}_aggregate_filtered.parquet')

    if not cfg[S_OVERWRITE_AGGREGATE] and not cfg[S_OVERWRITE_SUMMARY] and os.path.exists(cached_file):
        return pd.read_parquet(cached_file)
    FileUtils.silent_remove(cached_file)

    attr = DataAttributes(benchmark=cfg[S_BENCHMARK], node=node, category=S_CHARACTERIZATION,
                          overwrite_aggregate=cfg[S_OVERWRITE_AGGREGATE], overwrite_summary=False,
                          aggregate=True, metric=cfg[S_METRIC], return_as='json')

    attr.check_sample_size = False

    if S_KEEP_SUBMISSION_TIMESTAMP in cfg.keys():
        attr.keep_submission_timestamp = cfg[S_KEEP_SUBMISSION_TIMESTAMP]

    if cfg[S_BENCHMARK] in cfg[S_FILTER_BY_TS] and cfg[S_FILTER_BY_TS][cfg[S_BENCHMARK]] \
            and cfg[S_BENCHMARK] in cfg[S_SELECTED_TS].keys():
        attr.filter_by_timestamp = True
        attr.selected_timestamps = cfg[S_SELECTED_TS][cfg[S_BENCHMARK]]

    # df = ispass_22.get_benchmark_data_from_attribute(attr)

    df = sec_23.get_benchmark_data_from_attribute(attr)

    # df = sec_23_no_va_random.get_benchmark_data_from_attribute(attr)
    df.to_parquet(cached_file)
    return df


@dask.delayed
def par_calculate_and_set_group_median(external_group_df: pd.DataFrame, group_col=S_MEDIAN_OF_ALL):
    """ A dask method to set the median of medians on a per group (frequency configuration) basis.
    There is an assumption that parallelization is done at the benchmark level, so no need to group on
    benchmark
    keyword here. """
    results = []
    for group_name, group_df in external_group_df.groupby(S_DVFS_COLS):
        group_df[group_col] = group_df[S_SAMPLE_MEDIAN].median()
        results.append(group_df)
    return pd.concat(results)


@dask.delayed
def par_set_node_id(df: pd.DataFrame, node_id):
    """ A dask method to convert node-id from string to int. Runs in parallel on multiple partitions"""
    df[S_NODE_ID] = node_id_to_int(node_id)
    return df


def dask_remove_variability_outliers(df: pd.DataFrame):
    df = df.explode(column=[S_INTRA, S_INTER])
    return df


def mapped_calculate_inter_and_intra_variability(df: pd.DataFrame, metric_key, absolute_diff):
    """ A dask method that we use to map the calculation of inter, and intra node variability values on a per
    observation basis. This method gets called by the map method from dask API on a partitioned data frame."""
    df = df.drop(columns=[i for i in df.columns if i not in [S_INTRA, S_INTER, S_NODE_ID, S_SAMPLE_MEDIAN,
                                                             S_MEDIAN_OF_ALL, metric_key] + S_DVFS_COLS])
    df[S_INTRA] = df.apply(lambda x: Metrics.relative_error_percentage(x[S_SAMPLE_MEDIAN], x[metric_key],
                                                                       absolute=absolute_diff), axis=1)
    df[S_INTER] = df.apply(lambda x: Metrics.relative_error_percentage(x[S_MEDIAN_OF_ALL], x[metric_key],
                                                                       absolute=absolute_diff), axis=1)
    df = df.drop(columns=[S_MEDIAN_OF_ALL, metric_key, S_SAMPLE_MEDIAN])
    return df


def get_benchmark_data_for_nodes(nodes: list, cfg: dict, get_bm_data_method, metric_key, min_num_obs):
    """ The main method that gets the benchmarks data, and apply the subsequent transformations. """
    results = []
    for node in nodes:

        df = par_get_benchmark_data_for_node(node, cfg, get_bm_data_method)

        df = par_remove_metric_outliers(df, metric_key)

        # Select min number of observations
        if min_num_obs is not None:
            df = par_select_min_num_obs(df, metric_key, min_num_obs)

        df = par_calculate_sample_median(df, metric_key)

        # Set node ID to data records
        df = par_set_node_id(df, node)
        results.append(df)

    return results
