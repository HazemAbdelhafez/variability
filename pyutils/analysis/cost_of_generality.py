import os
from os.path import join as jp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyutils.characterization.networks.utils import get_print_benchmark_name
from pyutils.common.paths import FIGURES_DIR
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_NODE_ID, S_NETWORK, P_NETWORK, S_DVFS_CONFIG, S_PREDICTED_RUNTIME, \
    S_PREDICTED_ENERGY
from pyutils.modeling.helpers import S_NW_INFERENCE_TIME, S_NW_INFERENCE_ENERGY, S_NW_INFERENCE_ENERGY_PREDICTION_ERR, \
    S_NW_INFERENCE_RUNTIME_PREDICTION_ERR
from pyutils.modeling.kernels.kernels_models_prediction import NetworkPredictions, CACHING_DVFS_FILE
from pyutils.modeling.networks.networks_models_prediction import HighLevelNetworkPredictions


class CostOfGenerality:

    @staticmethod
    def load_kernels_based_models_results(overwrite=False):
        return NetworkPredictions.main(overwrite=overwrite, dvfs_file=CACHING_DVFS_FILE)

    @staticmethod
    def load_networks_base_models_results(overwrite=False):
        return HighLevelNetworkPredictions.main(overwrite=overwrite)

    @staticmethod
    def join_tables(kernel_level_results: pd.DataFrame, network_level_results):
        kept_cols = [S_NETWORK, S_DVFS_CONFIG, S_NODE_ID, S_NW_INFERENCE_TIME, S_NW_INFERENCE_ENERGY,
                     S_PREDICTED_RUNTIME, S_PREDICTED_ENERGY,
                     S_NW_INFERENCE_RUNTIME_PREDICTION_ERR, S_NW_INFERENCE_ENERGY_PREDICTION_ERR]
        kernel_level_results.drop(columns=[i for i in kernel_level_results.columns if i not in kept_cols], inplace=True)
        network_level_results.drop(columns=[i for i in network_level_results.columns if i not in kept_cols],
                                   inplace=True)

        kernel_level_results['Level'] = 'Kernel'
        network_level_results['Level'] = 'Network'

        df = pd.concat([kernel_level_results, network_level_results])
        return df

    @staticmethod
    def plot_runtime(df: pd.DataFrame):
        figures_dir = jp(FIGURES_DIR, 'cost-of-generality')
        os.makedirs(figures_dir, exist_ok=True)
        figure_path = jp(figures_dir, 'cost_of_generality.png')

        # We do not need frequency configurations or node id
        df.rename(columns={S_NETWORK: P_NETWORK}, inplace=True)
        df.drop(columns=[S_DVFS_CONFIG, S_NODE_ID], inplace=True)

        # Rename networks to pretty format
        df[P_NETWORK] = df[P_NETWORK].apply(lambda x: get_print_benchmark_name(x))

        value_name = 'Runtime Prediction Error %'
        df.rename(columns={S_NW_INFERENCE_RUNTIME_PREDICTION_ERR: value_name}, inplace=True)

        PlottingHelpers.set_fonts(small=12, medium=16, big=18)
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        var_name = 'Level'
        df.sort_values(by=[P_NETWORK], inplace=True)
        sns.boxplot(y=value_name, x=P_NETWORK, data=df, hue=var_name, showfliers=True,
                    ax=ax, whis=[10, 90])
        new_handles = PlottingHelpers.set_to_grey_scale(ax, hue_data=df[var_name].unique())
        _, legend_labels = ax.get_legend_handles_labels()

        ax.legend(new_handles, legend_labels, bbox_to_anchor=(0.5, 1.05), loc='center', ncol=3,
                  frameon=False, handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)
        ax.yaxis.grid(True)
        PlottingHelpers.rotate_x_labels(ax, angle=20)
        # Remove the default SNS x-label
        ax.setup(xlabel=None)
        plt.subplots_adjust(wspace=0.3, hspace=0.01)
        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=900)

    @staticmethod
    def plot_both_metrics(df: pd.DataFrame):
        figures_dir = jp(FIGURES_DIR, 'cost-of-generality')
        os.makedirs(figures_dir, exist_ok=True)
        figure_path = jp(figures_dir, 'cost_of_generality_runtime_and_energy.png')

        # We do not need frequency configurations or node id
        df.rename(columns={S_NETWORK: P_NETWORK}, inplace=True)
        df.drop(columns=[S_DVFS_CONFIG, S_NODE_ID], inplace=True)

        # Rename networks to pretty format
        df[P_NETWORK] = df[P_NETWORK].apply(lambda x: get_print_benchmark_name(x))

        PlottingHelpers.set_fonts(small=12, medium=16, big=18)
        fig, axs = plt.subplots(2, 1, figsize=(11, 10))
        runtime_ax = axs[0]
        energy_ax = axs[1]
        var_name = 'Level'
        df.sort_values(by=[P_NETWORK], inplace=True)

        value_name = 'Runtime Prediction Error %'
        df.rename(columns={S_NW_INFERENCE_RUNTIME_PREDICTION_ERR: value_name}, inplace=True)
        sns.boxplot(y=value_name, x=P_NETWORK, data=df, hue=var_name, showfliers=True, ax=runtime_ax,
                    whis=[10, 90])

        value_name = 'Energy Prediction Error %'
        df.rename(columns={S_NW_INFERENCE_ENERGY_PREDICTION_ERR: value_name}, inplace=True)
        sns.boxplot(y=value_name, x=P_NETWORK, data=df, hue=var_name, showfliers=True,
                    ax=energy_ax, whis=[10, 90])

        _ = PlottingHelpers.set_to_grey_scale(energy_ax, hue_data=df[var_name].unique())
        new_handles = PlottingHelpers.set_to_grey_scale(runtime_ax, hue_data=df[var_name].unique())
        _, legend_labels = runtime_ax.get_legend_handles_labels()

        runtime_ax.legend(new_handles, legend_labels, bbox_to_anchor=(0.5, 1.05), loc='center', ncol=3,
                          frameon=False, handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)
        energy_ax.legend([], frameon=False)

        runtime_ax.yaxis.grid(True)
        energy_ax.yaxis.grid(True)

        PlottingHelpers.rotate_x_labels(energy_ax, angle=20)
        PlottingHelpers.rotate_x_labels(runtime_ax, angle=20)
        # Remove the default SNS x-label
        runtime_ax.setup(xlabel=None)
        energy_ax.setup(xlabel=None)
        plt.subplots_adjust(wspace=0.3, hspace=0.18)
        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=900)

    @staticmethod
    def main(overwrite=False):
        k_data = CostOfGenerality.load_kernels_based_models_results(overwrite)
        n_data = CostOfGenerality.load_networks_base_models_results(overwrite)
        joined = CostOfGenerality.join_tables(k_data, n_data)
        CostOfGenerality.plot_both_metrics(joined)


if __name__ == '__main__':
    CostOfGenerality.main(overwrite=False)
