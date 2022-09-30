import os
import sys
from os.path import join as join_path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from pyutils.analysis.common.error_metrics import Metrics
from pyutils.characterization.networks.utils import VISION_NETWORKS as NETWORKS, get_print_benchmark_name
from pyutils.common.config import N_CPU_CORES, S_DVFS_COLS
from pyutils.common.data_interface import DataAttributes, DataHandler
from pyutils.common.experiments_utils import parse_dvfs_configs
from pyutils.common.methods import is_runtime, to_latex, is_power
from pyutils.common.paths import PREDICTION_FIGURES_DIR, NETWORKS_PREDICTION_DIR
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_NODE_ID, S_NETWORK, P_NETWORK, S_RUNTIME_MS, S_POWER, S_DVFS_CONFIG, S_AVG_PWR_W, \
    S_PREDICTED_RUNTIME, S_PREDICTED_ENERGY, S_PREDICTED_POWER, \
    S_CHARACTERIZATION
from pyutils.common.utils import GlobalLogger, FileUtils
from pyutils.hosts.agx import DVFSHelpers
from pyutils.hosts.common import HOSTNAME
from pyutils.modeling.helpers import ModelLoader, ModelSelectionFilter
from pyutils.modeling.helpers import S_NW_INFERENCE_TIME, S_NW_INFERENCE_ENERGY, S_NW_INFERENCE_ENERGY_PREDICTION_ERR, \
    S_NW_INFERENCE_RUNTIME_PREDICTION_ERR, PredictionHelpers, S_NW_INFERENCE_POWER, S_NW_INFERENCE_POWER_PREDICTION_ERR

logger = GlobalLogger().get_logger()

S_PRED_ERROR_PERCENTAGE = 'Error %'

DVFS_FILE = 'dvfs_3150.json'

# Cached models are loaded as globals only when needed
runtime_prediction_models_cache = dict()
power_prediction_models_cache = dict()


def predict_network_metric(nw, metric, dvfs_configs):
    # Prepare prediction input
    prediction_input = pd.DataFrame(dvfs_configs)

    if is_power(metric):
        power_label = S_PREDICTED_POWER
        power_model = power_prediction_models_cache.get(nw)
        power_prediction = power_model.predict(prediction_input)
        prediction_input[power_label] = power_prediction
    elif is_runtime(metric):
        runtime_label = S_PREDICTED_RUNTIME
        runtime_model = runtime_prediction_models_cache.get(nw)
        runtime_prediction = runtime_model.predict(prediction_input)
        prediction_input[runtime_label] = runtime_prediction
    else:
        raise Exception("Unknown metric.")

    return prediction_input


class HighLevelNetworkPredictions:
    """ Evaluate the prediction made by the network-level model"""
    saved_data_file = join_path(NETWORKS_PREDICTION_DIR, 'networks_prediction_with_high_level_models.csv')

    @staticmethod
    def estimates_versus_ground_truth(nws, node, metric, parallel_backend=None):
        """ This method collects the inference time of all networks, and it estimates the networks inference time using
        the network-level models and summarize the data in a dataframe"""

        def internal_method(_nw, _node):
            # If we didn't characterize a DVFS config, we return a None placeholder instead of exception.
            nw_data_attr = DataAttributes(node=_node, category=S_CHARACTERIZATION, benchmark=_nw,
                                          return_placeholder=True,
                                          return_all_columns=True, overwrite_summary=True)
            if is_runtime(metric):
                nw_inference_metric_data = DataHandler.get_nw_runtime(nw_data_attr, dvfs_configs)
                nw_inference_metric_key = S_NW_INFERENCE_TIME
            elif is_power(metric):
                nw_inference_metric_data = DataHandler.get_nw_power(nw_data_attr, dvfs_configs)
                nw_inference_metric_key = S_NW_INFERENCE_POWER
            else:
                raise Exception(f"Unsupported metric. {metric}")

            if nw_inference_metric_data is not None:
                predicted_network_metric = predict_network_metric(_nw, metric, dvfs_configs)
            else:
                predicted_network_metric = None

            nw_inference_metric_data[S_DVFS_CONFIG] = nw_inference_metric_data[S_DVFS_COLS].apply(
                lambda row: DVFSHelpers.get_dvfs_config_level(row), axis=1)
            predicted_network_metric[S_DVFS_CONFIG] = predicted_network_metric[S_DVFS_COLS].apply(
                lambda row: DVFSHelpers.get_dvfs_config_level(row), axis=1)
            nw_inference_metric_data.drop(columns=S_DVFS_COLS, inplace=True)
            predicted_network_metric.drop(columns=S_DVFS_COLS, inplace=True)

            combined_prediction_and_labels = pd.merge(nw_inference_metric_data, predicted_network_metric,
                                                      on=[S_DVFS_CONFIG])
            combined_prediction_and_labels[S_NETWORK] = _nw

            if is_runtime(metric):
                runtime_label = DataHandler.get_metric_label(metric)
                combined_prediction_and_labels.rename(columns={runtime_label: nw_inference_metric_key}, inplace=True)
            elif is_power(metric):
                power_label = DataHandler.get_metric_label(metric)
                combined_prediction_and_labels.rename(columns={power_label: nw_inference_metric_key}, inplace=True)
            else:
                raise Exception("Unknown metric.")

            return combined_prediction_and_labels

        dvfs_configs = parse_dvfs_configs(target_file=DVFS_FILE)
        results = parallel_backend(delayed(internal_method)(nw, node) for nw in nws)
        df = pd.concat(results)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def add_prediction_errors(df: pd.DataFrame) -> pd.DataFrame:
        # Add runtime prediction error
        df[S_NW_INFERENCE_RUNTIME_PREDICTION_ERR] = \
            Metrics.relative_error_percentage(df[S_NW_INFERENCE_TIME], df[S_PREDICTED_RUNTIME], absolute=False)

        # Add energy prediction error
        df[S_NW_INFERENCE_ENERGY_PREDICTION_ERR] = \
            Metrics.relative_error_percentage(df[S_NW_INFERENCE_ENERGY], df[S_PREDICTED_ENERGY], absolute=False)

        # Add energy prediction error
        df[S_NW_INFERENCE_POWER_PREDICTION_ERR] = \
            Metrics.relative_error_percentage(df[S_NW_INFERENCE_POWER], df[S_PREDICTED_POWER], absolute=False)

        return df

    @staticmethod
    def run_predictions():
        nodes = ['node-02', 'node-13']
        nws = NETWORKS
        n_jobs = N_CPU_CORES if HOSTNAME != 'xps' else 1

        # Aggregate across testing nodes
        aggregate_dfs = list()
        with Parallel(n_jobs=min(len(nws), n_jobs)) as parallel:
            for node in nodes:
                runtime_df = HighLevelNetworkPredictions.estimates_versus_ground_truth(nws=nws, node=node,
                                                                                       metric=S_RUNTIME_MS,
                                                                                       parallel_backend=parallel)

                power_df = HighLevelNetworkPredictions.estimates_versus_ground_truth(nws=nws, node=node,
                                                                                     metric=S_POWER,
                                                                                     parallel_backend=parallel)

                # Create energy dataframe
                tmp = pd.merge(runtime_df, power_df, how='left', on=[S_NETWORK, S_DVFS_CONFIG])
                tmp[S_NW_INFERENCE_ENERGY] = tmp[S_NW_INFERENCE_TIME] * tmp[S_NW_INFERENCE_POWER] / 1e3
                tmp[S_PREDICTED_ENERGY] = tmp[S_PREDICTED_RUNTIME] * tmp[S_PREDICTED_POWER] / 1e3
                tmp[S_NODE_ID] = node
                aggregate_dfs.append(tmp)

        return pd.concat(aggregate_dfs)

    @staticmethod
    def main(overwrite=False):
        if os.path.exists(HighLevelNetworkPredictions.saved_data_file) and not overwrite:
            return pd.read_csv(HighLevelNetworkPredictions.saved_data_file)

        # TODO: fix this selection criteria
        selection_filter = ModelSelectionFilter()
        ModelLoader.load_all_networks_models(S_RUNTIME_MS, runtime_prediction_models_cache, selection_filter)
        ModelLoader.load_all_networks_models(S_AVG_PWR_W, power_prediction_models_cache, selection_filter)

        for nw in NETWORKS:
            meta_file = ModelLoader.get_trained_model_meta_path(nw, label=S_RUNTIME_MS,
                                                                selection_filter=selection_filter)
            contents = FileUtils.deserialize(meta_file)
            if len(contents['testing_nodes']) != 2:
                logger.warning("Something is wrong with the networks models.")
                sys.exit(-1)

        df = HighLevelNetworkPredictions.run_predictions()
        with_errors = HighLevelNetworkPredictions.add_prediction_errors(df)
        with_errors.to_csv(HighLevelNetworkPredictions.saved_data_file, index=False)
        return with_errors


class HighLevelNetworkPredictionsPlots:

    @staticmethod
    def plot_relative_error(df: pd.DataFrame, figure_path='networks_relative_error.png'):
        # We remove the lowest frequency because of the high errors due to CPU vs GPU activity timing.
        df.drop(inplace=True, index=df[df[S_DVFS_CONFIG] == 'low'].index)
        # --------------------------------------------------------------------------------------------------------------
        # Drop non-error columns that we do not need in the plotting
        # --------------------------------------------------------------------------------------------------------------
        left_out_columns = list()
        for col in df.columns:
            if 'Error' not in str(col) and str(col) not in [S_NETWORK]:
                left_out_columns.append(col)
        df.drop(columns=left_out_columns, inplace=True)
        # **************************************************************************************************************

        # --------------------------------------------------------------------------------------------------------------
        # Rename the data columns for plotting
        # --------------------------------------------------------------------------------------------------------------
        renamed_cols = dict()
        for col in df.columns:
            original_col_name = str(col)
            new_col_name = str(col)
            if PredictionHelpers.is_runtime_data(original_col_name):
                new_col_name = new_col_name.replace(f"{S_NW_INFERENCE_TIME} vs ", "").lstrip().rstrip()
                new_col_name = new_col_name.replace("Relative Error %", "").lstrip().rstrip()
                if new_col_name == S_PREDICTED_RUNTIME:
                    new_col_name = 'Selected Kernels Predicted Runtime'
            elif PredictionHelpers.is_energy_data(original_col_name):
                new_col_name = new_col_name.replace(f"{S_NW_INFERENCE_ENERGY} vs ", "").lstrip().rstrip()
                new_col_name = new_col_name.replace("Relative Error %", "").lstrip().rstrip()
                if new_col_name == S_PREDICTED_ENERGY:
                    new_col_name = 'Selected Kernels Predicted Energy'
            else:
                pass
            renamed_cols[original_col_name] = new_col_name
        df = df.rename(columns=renamed_cols)
        # **************************************************************************************************************

        # --------------------------------------------------------------------------------------------------------------
        # Stack the DF
        # --------------------------------------------------------------------------------------------------------------
        var_name = 'Baseline'
        value_name = 'Error %'
        stacked_df = df.melt(id_vars=[S_NETWORK], var_name=var_name, value_name=value_name)
        # Split the energy and runtime rows
        runtime_df = stacked_df[stacked_df[var_name].str.contains('Runtime')]
        energy_df = stacked_df[stacked_df[var_name].str.contains('Energy')]
        logger.info(f"Average runtime absolute relative error: "
                    f"{runtime_df[runtime_df['Baseline'] == 'Selected Kernels Predicted Runtime']['Error %'].abs().mean()}")
        logger.info(f"Average energy absolute relative error: "
                    f"{energy_df[energy_df['Baseline'] == 'Selected Kernels Predicted Energy']['Error %'].abs().mean()}")

        # **************************************************************************************************************

        # --------------------------------------------------------------------------------------------------------------
        # Create and plot the figures
        # --------------------------------------------------------------------------------------------------------------
        fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [9, 1]}, figsize=(21, 7))
        runtime_ax = axs[0, 0]
        energy_ax = axs[1, 0]

        shufflenet_runtime_ax = axs[0, 1]
        shufflenet_energy_ax = axs[1, 1]

        no_shufflenet_runtime_df = runtime_df[runtime_df[S_NETWORK] != 'shufflenetv2']
        sns.boxplot(y=value_name, x=S_NETWORK, data=no_shufflenet_runtime_df, hue=var_name, showfliers=True,
                    ax=runtime_ax, whis=[10, 90])

        no_shufflenet_energy_df = energy_df[energy_df[S_NETWORK] != 'shufflenetv2']
        sns.boxplot(y=value_name, x=S_NETWORK, data=no_shufflenet_energy_df, hue=var_name, showfliers=True,
                    ax=energy_ax, whis=[10, 90])

        shufflenet_runtime_df = runtime_df[runtime_df[S_NETWORK] == 'shufflenetv2']
        sns.boxplot(y=value_name, x=S_NETWORK, data=shufflenet_runtime_df, hue=var_name, showfliers=True,
                    ax=shufflenet_runtime_ax, whis=[10, 90])

        shufflenet_energy_df = energy_df[energy_df[S_NETWORK] == 'shufflenetv2']
        sns.boxplot(y=value_name, x=S_NETWORK, data=shufflenet_energy_df, hue=var_name, showfliers=True,
                    ax=shufflenet_energy_ax, whis=[10, 90])

        # Set y-labels
        runtime_ax.set_ylabel('Runtime Error %')
        energy_ax.set_ylabel('Energy Error %')

        PlottingHelpers.remove_y_axis_label([shufflenet_energy_ax, shufflenet_runtime_ax])

        # Remove the x-label and ticks for the first row sub-figure
        PlottingHelpers.remove_x_axis_labels_and_ticks([runtime_ax, shufflenet_runtime_ax])

        # Capitalize the networks names in the second row sub-figure
        PlottingHelpers.capitalize_x_axis_labels([energy_ax, shufflenet_energy_ax])

        # Turn box plots to grey scale for all sub-plots
        new_handles = PlottingHelpers.set_to_grey_scale(runtime_ax, hue_data=runtime_df[var_name].unique())
        PlottingHelpers.set_to_grey_scale(energy_ax, hue_data=energy_df[var_name].unique())
        PlottingHelpers.set_to_grey_scale(shufflenet_runtime_ax, hue_data=shufflenet_runtime_df[var_name].unique())
        PlottingHelpers.set_to_grey_scale(shufflenet_energy_ax, hue_data=shufflenet_energy_df[var_name].unique())

        # We remove the bottom sub-figure legend
        PlottingHelpers.remove_legend([energy_ax, shufflenet_energy_ax, shufflenet_runtime_ax])

        # Set legend boxes to use the new colors:
        _, labels = runtime_ax.get_legend_handles_labels()
        labels = [
            str(j).replace(" Runtime", '').replace('Predicted', 'Predictions')
            for j in labels]
        runtime_ax.legend(new_handles, labels, bbox_to_anchor=(0.6, 1.07), loc='center', ncol=3, frameon=False,
                          handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)

        # Set grid lines for all axes
        runtime_ax.yaxis.grid(True)
        shufflenet_runtime_ax.yaxis.grid(True)
        energy_ax.yaxis.grid(True)
        shufflenet_energy_ax.yaxis.grid(True)

        # Remove the default SNS xlabel
        runtime_ax.setup(xlabel=None)
        shufflenet_runtime_ax.setup(xlabel=None)
        energy_ax.setup(xlabel=None)
        shufflenet_energy_ax.setup(xlabel=None)

        # Move right plots y-axis to right
        PlottingHelpers.move_y_axis_to_right([shufflenet_runtime_ax, shufflenet_energy_ax])

        # Set plots whitespace sizes
        plt.subplots_adjust(wspace=0.008, hspace=0.005)
        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=600)
        # **************************************************************************************************************

    @staticmethod
    def print_report(overwrite_data=False):
        # Load saved data file, or create it
        df = HighLevelNetworkPredictions.main(overwrite=overwrite_data)
        keep_columns = [S_NETWORK,
                        S_NW_INFERENCE_TIME, S_PREDICTED_RUNTIME,
                        S_NW_INFERENCE_ENERGY, S_PREDICTED_ENERGY,
                        S_NW_INFERENCE_POWER, S_PREDICTED_POWER]

        df.drop(columns=[i for i in df.columns if i not in keep_columns], inplace=True)

        runtime_df = df[[S_NETWORK, S_NW_INFERENCE_TIME, S_PREDICTED_RUNTIME]].copy()
        power_df = df[[S_NETWORK, S_NW_INFERENCE_POWER, S_PREDICTED_POWER]].copy()
        energy_df = df[[S_NETWORK, S_NW_INFERENCE_ENERGY, S_PREDICTED_ENERGY]].copy()

        logger.info(f"Average runtime absolute relative error: "
                    f"{Metrics.relative_error_percentage(df[S_NW_INFERENCE_TIME], df[S_PREDICTED_RUNTIME], absolute=True).mean()}")
        logger.info(f"Average power absolute relative error: "
                    f"{Metrics.relative_error_percentage(power_df[S_NW_INFERENCE_POWER], power_df[S_PREDICTED_POWER], absolute=True).mean()}")
        logger.info(f"Average energy absolute relative error: "
                    f"{Metrics.relative_error_percentage(energy_df[S_NW_INFERENCE_ENERGY], energy_df[S_PREDICTED_ENERGY], absolute=True).mean()}")

        # Format runtime data
        runtime_data = list()
        for name, nw_data in runtime_df.groupby(by=[S_NETWORK]):
            results = Metrics.combined_prediction_evaluation_metrics(nw_data[S_NW_INFERENCE_TIME],
                                                                     nw_data[S_PREDICTED_RUNTIME],
                                                                     as_type='pretty_dict')
            tmp = dict()
            tmp[P_NETWORK] = get_print_benchmark_name(name)
            tmp.update(results)
            tmp['RMSE (ms)'] = tmp.pop('RMSE')
            tmp['MAE (ms)'] = tmp.pop('MAE')
            runtime_data.append(tmp)
        runtime_df = pd.DataFrame(runtime_data).round(2)

        # Append mean row
        mean_row = dict()
        for col in runtime_df.columns:
            if col == P_NETWORK:
                mean_row[col] = 'Mean'
            else:
                mean_row[col] = runtime_df[col].mean()
        runtime_df = runtime_df.append(pd.Series(mean_row), ignore_index=True)
        runtime_df = runtime_df.round(2)

        print("Runtime")
        latex_code = to_latex(runtime_df)
        print(latex_code)

        # Format energy data
        energy_data = list()

        for name, nw_data in energy_df.groupby(by=[S_NETWORK]):
            results = Metrics.combined_prediction_evaluation_metrics(nw_data[S_NW_INFERENCE_ENERGY],
                                                                     nw_data[S_PREDICTED_ENERGY], as_type='pretty_dict')
            tmp = dict()
            tmp[P_NETWORK] = get_print_benchmark_name(name)
            tmp.update(results)
            tmp['RMSE (J)'] = tmp.pop('RMSE')
            tmp['MAE (J)'] = tmp.pop('MAE')
            energy_data.append(tmp)
        energy_df = pd.DataFrame(energy_data)

        # Append mean row
        mean_row = dict()
        for col in energy_df.columns:
            if col == P_NETWORK:
                mean_row[col] = 'Mean'
            else:
                mean_row[col] = energy_df[col].mean()
        energy_df = energy_df.append(pd.Series(mean_row), ignore_index=True)
        energy_df['5%'] = energy_df['5%'].round(2)
        energy_df['10%'] = energy_df['10%'].round(2)
        energy_df['15%'] = energy_df['15%'].round(2)
        energy_df['20%'] = energy_df['20%'].round(2)
        energy_df['RMSE (J)'] = energy_df['RMSE (J)'].round(3)
        energy_df['MAE (J)'] = energy_df['MAE (J)'].round(3)

        # Format power data
        power_data = list()

        for name, nw_data in power_df.groupby(by=[S_NETWORK]):
            results = Metrics.combined_prediction_evaluation_metrics(nw_data[S_NW_INFERENCE_POWER],
                                                                     nw_data[S_PREDICTED_POWER], as_type='pretty_dict')
            tmp = dict()
            tmp[P_NETWORK] = get_print_benchmark_name(name)
            tmp.update(results)
            tmp['RMSE (W)'] = tmp.pop('RMSE')
            tmp['MAE (W)'] = tmp.pop('MAE')
            power_data.append(tmp)
        power_df = pd.DataFrame(power_data)

        # Append mean row
        mean_row = dict()
        for col in power_df.columns:
            if col == P_NETWORK:
                mean_row[col] = 'Mean'
            else:
                mean_row[col] = power_df[col].mean()
        power_df = power_df.append(pd.Series(mean_row), ignore_index=True)
        power_df = power_df.round(2)

        print("Power")
        latex_code = to_latex(power_df)
        print(latex_code)

        print("Energy")
        latex_code = to_latex(energy_df)
        print(latex_code)

    @staticmethod
    def main(overwrite_data=False, overwrite_figures=False):
        if overwrite_data:
            DataHandler.clear_cached_summary(category=S_CHARACTERIZATION)

        # Load saved data file, or create it
        if not os.path.exists(HighLevelNetworkPredictions.saved_data_file) or overwrite_data:
            HighLevelNetworkPredictions.main(overwrite=overwrite_data)

        figure_name = 'networks_relative_error.png'
        os.makedirs(PREDICTION_FIGURES_DIR, exist_ok=True)
        figure_path = join_path(PREDICTION_FIGURES_DIR, figure_name)
        if not os.path.exists(figure_path) or overwrite_figures:
            df = pd.read_csv(HighLevelNetworkPredictions.saved_data_file)
            PlottingHelpers.set_fonts(small=12, medium=16, big=20)
            HighLevelNetworkPredictionsPlots.plot_relative_error(df, figure_path)


if __name__ == '__main__':
    HighLevelNetworkPredictionsPlots.print_report(overwrite_data=False)
    # HighLevelNetworkPredictions.main(overwrite=True)
    # TODO: the plots methods are copied from the kernels module, and not used for now.
    # NetworkPredictionsPlots.main(overwrite_data=True, overwrite_figures=True)
