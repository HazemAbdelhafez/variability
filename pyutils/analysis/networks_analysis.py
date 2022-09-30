from os.path import exists
from os.path import join as jp

import pandas as pd
from pyutils.analysis.analyze_profiling_traces import RuntimeAnalyzer

from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.networks.utils import VISION_NETWORKS, get_print_benchmark_name
from pyutils.common.config import THIRTEEN_NODES
from pyutils.common.data_interface import DataHandler, DataAttributes
from pyutils.common.experiments_utils import parse_dvfs_configs
from pyutils.common.methods import to_latex
from pyutils.common.paths import VISION_NETWORKS_PROFILING_OUTPUT_DIR, NETWORKS_CHARACTERIZATION_DIR
from pyutils.common.strings import S_CONV, S_BATCHNORM, S_ADD, S_RELU, S_MAXPOOL, \
    S_MATMUL, \
    S_ADAPTIVEPOOL, S_CAT
from pyutils.common.strings import S_NODE_ID, S_NETWORK, P_NETWORK, S_RUNTIME_MS, S_DVFS_CONFIG, S_KERNEL, S_SUPPORTED, \
    S_CHARACTERIZATION
from pyutils.hosts.agx import DVFSHelpers
from pyutils.modeling.helpers import S_NW_INFERENCE_TIME
from pyutils.modeling.kernels.kernels_models_prediction import S_SUM_OF_SUPPORTED_KERNELS_RUNTIME, \
    S_SUM_OF_ALL_KERNELS_RUNTIME, S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME


class CharacterizeNetworks:
    """ This class allows us to study the CNNs and extract the main kernels using
    data collected from multiple boards"""
    networks_characterization_data_file = jp(NETWORKS_CHARACTERIZATION_DIR, 'networks-characterization.csv')
    dvfs_file = 'dvfs_3_1.json'

    @staticmethod
    def get_kernel_timing_data(nw, kernel, dvfs_config, node):
        """ This method gets a kernel timing data as a dict of: kernel measured runtime, and whether
        the kernel is a supported one or not, for a specific node and DVFS configuration. """

        # Extract kernel information
        kernel_name = KernelsChecks.get_unified_kernel_name(list(kernel.keys())[0])
        kernel_parameters = list(kernel.values())[0]

        # Get kernel runtime
        data_attr = DataAttributes(node=node, benchmark=nw, kernel=kernel_name, return_placeholder=True)
        kernel_runtime = DataHandler.get_kernel_runtime(data_attr, dvfs_configs=dvfs_config,
                                                        kernel_parameters=kernel_parameters.to_dict())
        kernel_data = {S_KERNEL: kernel_name,
                       S_RUNTIME_MS: kernel_runtime}

        return kernel_data

    @staticmethod
    def extract_data(overwrite_data=False):
        saved_data_file = CharacterizeNetworks.networks_characterization_data_file

        if exists(saved_data_file) and not overwrite_data:
            return pd.read_csv(saved_data_file)

        dvfs_configs = parse_dvfs_configs(target_file=CharacterizeNetworks.dvfs_file)

        data = list()
        for nw in VISION_NETWORKS:
            nw_dir = jp(VISION_NETWORKS_PROFILING_OUTPUT_DIR, nw)
            stack_file = jp(nw_dir, 'stack_content.csv')
            chrome_file = jp(nw_dir, f'{nw}_chrome.trace')

            # All kernels
            kernels = RuntimeAnalyzer.get_top_level_kernels(stack_file, chrome_file, supported_only=False)

            for node in THIRTEEN_NODES:
                nw_data_attr = DataAttributes(benchmark=nw, node=node, category=S_CHARACTERIZATION)

                for dvfs_config in dvfs_configs:
                    nw_inference_time = DataHandler.get_nw_runtime(nw_data_attr, dvfs_config)

                    kernels_data = list()
                    for kernel in kernels:
                        kernel_timing_data = \
                            CharacterizeNetworks.get_kernel_timing_data(nw=nw, kernel=kernel,
                                                                        dvfs_config=dvfs_config, node=node)
                        kernels_data.append(kernel_timing_data)

                    kernels_timing_data = pd.DataFrame(kernels_data)
                    # Group by and sum the runtime for each kernel, then sort by runtime in descending order
                    kernels_timing_data: pd.DataFrame = \
                        kernels_timing_data.groupby([S_KERNEL])[S_RUNTIME_MS].sum().reset_index()
                    kernels_timing_data[S_SUPPORTED] = kernels_timing_data[S_KERNEL].apply(
                        KernelsChecks.is_supported)
                    kernels_timing_data.sort_values(by=[S_RUNTIME_MS], inplace=True, ascending=False)

                    supported_kernels_runtime = \
                        kernels_timing_data[kernels_timing_data[S_SUPPORTED]][S_RUNTIME_MS].sum()
                    all_kernels_runtime = kernels_timing_data[S_RUNTIME_MS].sum()

                    # Drop supported flag, orient the df as such, we have a dictionary of kernels as keys, and runtime
                    # as values.
                    kernels_timing_data.drop(columns=[S_SUPPORTED], inplace=True)
                    kernels_timing_data.set_index(S_KERNEL, inplace=True)
                    kernels_runtime_dict = kernels_timing_data.to_dict()[S_RUNTIME_MS]

                    data.append({**{S_NETWORK: nw,
                                    S_DVFS_CONFIG: DVFSHelpers.get_dvfs_config_level(dvfs_config),
                                    S_NW_INFERENCE_TIME: nw_inference_time,
                                    S_SUM_OF_SUPPORTED_KERNELS_RUNTIME: supported_kernels_runtime,
                                    S_SUM_OF_ALL_KERNELS_RUNTIME: all_kernels_runtime,
                                    S_NODE_ID: node}, **kernels_runtime_dict})

        df = pd.DataFrame(data)
        df.to_csv(saved_data_file, index=False)
        return df

    @staticmethod
    def create_characterization_table(df: pd.DataFrame):
        # Remove unsupported kernels columns
        df.drop(columns=[i for i in df.columns if KernelsChecks.is_unsupported(i)], inplace=True)

        # Replace each kernel column by its percentage contribution to the inference time
        existing_kernels = [i for i in df.columns if KernelsChecks.is_kernel(i)]
        for kernel in existing_kernels:
            df[kernel] = df[kernel] * 100 / df[S_NW_INFERENCE_TIME]

        df.sort_values(by=[S_DVFS_CONFIG], inplace=True)

        return df

    @staticmethod
    def select_from_characterization_table(df: pd.DataFrame):
        # For the time being, we select to report the numbers at the highest
        # frequency configuration, otherwise, change the next line of code.
        df.drop(index=df[df[S_DVFS_CONFIG] != 'high'].index, inplace=True)
        df = df.groupby(by=[S_NETWORK], as_index=False).median()

        # Convert the all kernels measured runtime to left-out kernels runtime
        df[S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME] = df[S_SUM_OF_ALL_KERNELS_RUNTIME] - \
                                                   df[S_SUM_OF_SUPPORTED_KERNELS_RUNTIME]

        df.drop(columns=[S_SUM_OF_ALL_KERNELS_RUNTIME], inplace=True)

        return df

    @staticmethod
    def format_characterization_table(df: pd.DataFrame):

        # Add the last analysis columns
        df["\\frac{T_{k}}{T} (%)"] = (100 * df[S_SUM_OF_SUPPORTED_KERNELS_RUNTIME] / df[S_NW_INFERENCE_TIME]).round(2)
        df["\\frac{T_{k}}{T_{k} + T_{k'}} (%)"] = \
            (100 * df[S_SUM_OF_SUPPORTED_KERNELS_RUNTIME] /
             (df[S_SUM_OF_SUPPORTED_KERNELS_RUNTIME] + df[S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME])).round(2)
        df["\\frac{T_{k} + T_{k'}}{T} (%)"] = \
            (100 * (df[S_SUM_OF_SUPPORTED_KERNELS_RUNTIME] + df[S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME]) /
             df[S_NW_INFERENCE_TIME]).round(2)

        # Append mean row
        mean_row = dict()
        for col in df.columns:
            if col == S_NETWORK:
                mean_row[col] = 'Mean'
            else:
                mean_row[col] = df[col].mean()
        df = df.append(pd.Series(mean_row), ignore_index=True)

        # Set decimals
        df[S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME] = df[S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME].round(3)
        for col in df.columns:
            if col not in [S_NETWORK, S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME]:
                df[col] = df[col].round(2)

        # Rename columns
        renaming_dict = {S_NETWORK: P_NETWORK,
                         S_NW_INFERENCE_TIME: 'T (ms)',
                         S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME: "T_{k'} (ms)",
                         S_SUM_OF_SUPPORTED_KERNELS_RUNTIME: "T_{k} (ms)"
                         }
        for col in df.columns:
            if KernelsChecks.is_supported(col):
                renaming_dict.update({col: KernelsChecks.get_print_kernel_name(col)})

        df[S_NETWORK] = df[S_NETWORK].apply(get_print_benchmark_name)
        df.rename(columns=renaming_dict, inplace=True)

        # Sorted kernels listing by mean of their contribution percentage
        sorted_columns = [P_NETWORK]
        for k in [S_CONV, S_BATCHNORM, S_MATMUL, S_RELU, S_CAT, S_MAXPOOL, S_ADD, S_ADAPTIVEPOOL]:
            sorted_columns.append(KernelsChecks.get_print_kernel_name(k))
        sorted_columns += ["T_{k} (ms)", "T_{k'} (ms)", 'T (ms)']
        sorted_columns += [i for i in df.columns if i not in sorted_columns]
        df = df[sorted_columns]
        latex = to_latex(df, label='table:kernels_selection')
        print(latex)
        return df

    @staticmethod
    def main(overwrite_data=False):
        if overwrite_data:
            DataHandler.clear_cached_summary(category=S_CHARACTERIZATION)

        df = CharacterizeNetworks.extract_data(overwrite_data)
        df = CharacterizeNetworks.create_characterization_table(df)
        df = CharacterizeNetworks.select_from_characterization_table(df)
        CharacterizeNetworks.format_characterization_table(df)


if __name__ == '__main__':
    CharacterizeNetworks.main(overwrite_data=False)
