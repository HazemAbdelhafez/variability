from typing import Dict, List, Union

import pandas as pd

from pyutils.characterization.networks.utils import VISION_NETWORKS, get_print_benchmark_name
from pyutils.common.utils import create_cache
from pyutils.characterization.common.parameters import BaseParameters
from pyutils.characterization.networks.analyzers import stack_trace as nw_stack_trace
from pyutils.common.config import S_DVFS_COLS
from pyutils.common.data_handlers.data_interface import DataAttributes, DataHandler
from pyutils.common.methods import check_nan
from pyutils.common.strings import S_RUNTIME, S_KERNEL, S_RUNTIME_MS, S_CHARACTERIZATION, S_PARAMETERS, \
    S_BENCHMARK, S_DVFS_CONFIG
from pyutils.hosts.agx import DVFSHelpers


# Pandas display parameters
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_kernel_data(node, parameters: BaseParameters, metric: str, dvfs_configs: Union[Dict, List[Dict]]):
    """
        Reads the a kernel data file to parse its timing (or power) information. This kernel is called from
        a specific network during inference. So, this is not a generic method to extract kernels runtime.
        This is specific to extracting the constituent kernels of a specific network.
    Args:
        node: str, required
            The node on which we did the characterization and which to extract the metric data.
        parameters: BaseParameters, required
            The parameters of the kernel we are interested in.
        metric: str, required
            The metric to extract data for (runtime, power, or energy)
        dvfs_configs: Dict or List[Dict], required
            A single or multiple DVFS configs.

    Returns:
        r: Dataframe
            A dataframe object that contains the DVFS configs, kernel name, and summary value for the metric
             (default: median).

    """
    if isinstance(dvfs_configs, dict):
        dvfs_configs = [dvfs_configs]
    attr = DataAttributes(node=node, benchmark=parameters.name, metric=metric, return_as='df',
                          return_all_columns=True,
                          sub_characterization_type=S_KERNEL, overwrite_summary=False)
    r = DataHandler.get_kernel_runtime(attr, dvfs_configs, kernel_parameters=parameters.to_dict())
    r = r.drop(columns=[i for i in r.columns if i not in S_DVFS_COLS + [DataHandler.get_metric_key(metric),
                                                                        S_BENCHMARK]])
    return r


def get_network_kernels_runtime(nw, node="node-15", dvfs_config_file="dvfs_3.json") -> pd.DataFrame:
    """
    For a specific network, get its children kernels (i.e., called during inference), then extract their
    timing (or power) information for a particular set of DVFS configurations loaded from a specific file.
    Args:
        nw:
        node:
        dvfs_config_file:

    Returns:
        summary: Dataframe
            A dataframe object summarizing all kernels, for the specified DVFS configs, and the summary
            value for the metric of interest.

    """
    dvfs_configs = DVFSHelpers.load_file(dvfs_config_file)
    summary = list()
    kernels = nw_stack_trace.get_kernels(nw, supported_only=False)
    for name, parameters in kernels.items():
        for item in parameters:
            kernel_data = get_kernel_data(node, item, metric=S_RUNTIME, dvfs_configs=dvfs_configs)
            summary.append(kernel_data)
    summary = pd.concat(summary, ignore_index=True)
    return summary


def get_network_runtime(nw, node="node-15", dvfs_config_file="dvfs_3.json"):
    dvfs_configs = DVFSHelpers.load_file(dvfs_config_file)
    attr = DataAttributes(node=node, category=S_CHARACTERIZATION, benchmark=nw, return_all_columns=True,
                          overwrite_summary=False)
    runtime: pd.DataFrame = DataHandler.get_nw_runtime(attr, dvfs_configs)
    return runtime


def get_network_runtime_table(nw, node="node-15", dvfs_config_file="dvfs_3.json"):
    cache = create_cache(nw, node, dvfs_config_file, clear=True)
    if cache.valid():
        return cache.load()

    kernels_runtime_df = get_network_kernels_runtime(nw, node, dvfs_config_file)
    check_nan(kernels_runtime_df)
    kernels_runtime_df = kernels_runtime_df.drop(columns=[S_BENCHMARK]).groupby(by=S_DVFS_COLS).agg('sum')
    nw_runtime_df = get_network_runtime(nw)
    nw_runtime_df: pd.DataFrame = nw_runtime_df.drop(columns=[S_PARAMETERS])
    combined_df = nw_runtime_df.merge(kernels_runtime_df, how='left', on=S_DVFS_COLS,
                                      suffixes=("_nw", "_kernels"))
    combined_df["Ratio %"] = 100 * combined_df[f"{S_RUNTIME_MS}_kernels"]/combined_df[f"{S_RUNTIME_MS}_nw"]

    return cache.save(combined_df)


def get_networks_runtime_table():
    cache = create_cache(clear=True)
    if cache.valid():
        return cache.load()
    combined = list()
    for nw in VISION_NETWORKS:
        tmp = get_network_runtime_table(nw)
        combined.append(tmp)
    df = pd.concat(combined)
    return cache.save(df)


def formatted_report_v1(df: pd.DataFrame):
    for gn, gd in df.groupby(by=S_DVFS_COLS):
        tmp = gd.copy()
        tmp[S_BENCHMARK] = gd[S_BENCHMARK].apply(get_print_benchmark_name)
        tmp[S_DVFS_CONFIG] = gd[S_DVFS_COLS].apply(DVFSHelpers.get_dvfs_config_level, axis=1)
        tmp = tmp.drop(columns=S_DVFS_COLS)
        print(tmp.to_csv(index=False))


def main():
    df = get_networks_runtime_table()
    formatted_report_v1(df)


if __name__ == '__main__':
    # args_parser = argparse.ArgumentParser()
    # args_parser.add_argument("-c", "--config_file", required=True, help="")
    # cmd_line_args = vars(args_parser.parse_args())
    main()
