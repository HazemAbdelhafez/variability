import os
from os.path import join as join_path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from pyutils.analysis.common.error_metrics import Metrics
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.networks.analyzers.runtime import RuntimeAnalyzer
from pyutils.characterization.networks.utils import VISION_NETWORKS as NETWORKS, get_print_benchmark_name
from pyutils.common.config import N_CPU_CORES
from pyutils.common.data_handlers.data_interface import DataAttributes, DataHandler
from pyutils.common.experiments_utils import parse_dvfs_configs
from pyutils.common.methods import is_runtime, is_power, is_energy, to_latex
from pyutils.common.paths import PREDICTION_FIGURES_DIR, NETWORKS_PREDICTION_DIR
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_NODE_ID, S_NETWORK, P_NETWORK, S_RUNTIME_MS, S_RUNTIME, S_POWER, S_ENERGY, \
    S_ENERGY_J, \
    S_DVFS_CONFIG, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_AVG_PWR_W, S_PREDICTED_RUNTIME, S_PREDICTED_ENERGY, \
    S_KERNEL, S_SUPPORTED, S_CHARACTERIZATION
from pyutils.common.utils import GlobalLogger, FileUtils
from pyutils.hosts.agx import DVFSHelpers
from pyutils.hosts.common import HOSTNAME
from pyutils.modeling.helpers import S_NW_INFERENCE_TIME, S_NW_INFERENCE_ENERGY, S_NW_INFERENCE_ENERGY_PREDICTION_ERR, \
    S_NW_INFERENCE_RUNTIME_PREDICTION_ERR, PredictionHelpers, ModelSelectionFilter, TrainedModelWrapper
from pyutils.modeling.helpers import TrainingHelpers, ModelLoader

logger = GlobalLogger().get_logger()

CACHING_DVFS_FILE = 'dvfs_3150.json'
PLOTTING_DVFS_FILE = 'dvfs_3_1.json'

S_PRED_ERROR_PERCENTAGE = 'Error %'
S_SUM_OF_SUPPORTED_KERNELS_RUNTIME = 'Selected Kernels Observations Runtime'
S_SUM_OF_UNSUPPORTED_KERNELS_RUNTIME = 'Left-out Kernels Observations Runtime'
S_SUM_OF_SUPPORTED_KERNELS_ENERGY = 'Selected Kernels Observations Energy'
S_SUM_OF_ALL_KERNELS_RUNTIME = 'All Kernels Observations Runtime'
S_SUM_OF_ALL_KERNELS_ENERGY = 'All Kernels Observations Energy'

# Cached models are loaded as globals only when needed
runtime_prediction_models_cache = dict()
power_prediction_models_cache = dict()

runtime_prediction_files_cache = dict()
power_prediction_files_cache = dict()


def cache_predictions(kernels, dvfs_configs, metric, model_dir, overwrite=False):
    # Cached files should take the same directory structure as the saved models, if not the same folder.
    # Extract kernel information
    kernel_name = KernelsChecks.get_unified_kernel_name(list(kernels[0].keys())[0])
    if not KernelsChecks.is_supported(kernel_name):
        return

    # Load the training model
    training_model = TrainingHelpers.load_kernel_model(kernel_name)

    cached_file_path = join_path(model_dir, f'{kernel_name}_cached_predictions.csv')

    if overwrite or not os.path.exists(cached_file_path):
        FileUtils.silent_remove(cached_file_path)

    if os.path.exists(cached_file_path):
        logger.info("Loading pre-existing file")
        tmp = pd.read_csv(cached_file_path)
        logger.info(f"Cache contains: {tmp.shape} data")
        return

    # Load prediction model
    if is_runtime(metric):
        prediction_model = runtime_prediction_models_cache.get(kernel_name)
    elif is_power(metric):
        prediction_model = power_prediction_models_cache.get(kernel_name)
    else:
        raise Exception(f"Unknown metric")

    # Combined predictions container
    combined_predictions = list()

    for kernel in kernels:

        kernel_parameters = list(kernel.values())[0]

        # Preprocess the kernel parameters data
        kernel_parameters = training_model.preprocess_kernel_parameters(kernel_parameters)

        for dvfs_config in dvfs_configs:
            # Prepare prediction input
            combined_predictions.append({**kernel_parameters, **dvfs_config})

    cached_records = pd.DataFrame(combined_predictions)

    if is_runtime(metric):
        cached_records[S_RUNTIME_MS] = prediction_model.predict(cached_records)
    elif is_power(metric):
        cached_records[S_AVG_PWR_W] = prediction_model.predict(cached_records)

    # Cache the file path to the global dictionary
    if is_runtime(metric):
        runtime_prediction_files_cache[kernel_name] = \
            runtime_prediction_files_cache.get(kernel_name, cached_file_path)
    elif is_power(metric):
        power_prediction_files_cache[kernel_name] = \
            power_prediction_files_cache.get(kernel_name, cached_file_path)

    # Save to file
    logger.info(f"Save to: {cached_file_path}")
    cached_records.to_csv(cached_file_path, index=False)


def load_cached_prediction(kernel, dvfs_configs, cached_file_path):
    kernel_name = KernelsChecks.get_unified_kernel_name(list(kernel.keys())[0])
    kernel_parameters = list(kernel.values())[0]

    # Load the training model
    training_model = TrainingHelpers.load_kernel_model(kernel_name)

    # Preprocess the kernel parameters data
    kernel_parameters = training_model.preprocess_kernel_parameters(kernel_parameters)

    # Prepare prediction input
    combined_records = list()
    if type(dvfs_configs) is not list:
        dvfs_configs = [dvfs_configs]

    for config in dvfs_configs:
        combined_records.append({**kernel_parameters, **config})
    prediction_input = pd.DataFrame(combined_records)
    cached_records = pd.read_csv(cached_file_path)
    intersection = pd.merge(cached_records, prediction_input, how='inner', on=[i for i in prediction_input.columns])
    return intersection


def get_kernel_metric_data(nw, node, kernel, metric, dvfs_configs):
    """ This method gets a kernel timing data as a dict of: kernel measured runtime, predicted runtime, and whether
    the kernel is a supported one or not, for a specific node and DVFS configuration. """

    # Extract kernel information
    kernel_name = KernelsChecks.get_unified_kernel_name(list(kernel.keys())[0])
    kernel_parameters = list(kernel.values())[0]

    if is_runtime(metric):
        prediction_key = S_PREDICTED_RUNTIME
    elif is_energy(metric):
        prediction_key = S_PREDICTED_ENERGY
    else:
        raise Exception("Unknown metric")

    # TODO: ugly handling of different DVFS files depending on whether we plot or report tables
    is_plot = len(dvfs_configs) == 3

    if is_plot:
        # Get kernel runtime
        data_attr = DataAttributes(node=node, benchmark=nw, kernel=kernel_name, return_placeholder=True,
                                   return_all_columns=True, overwrite_summary=True)
        if is_runtime(metric):
            kernel_metric_data = DataHandler.get_kernel_runtime(data_attr, dvfs_configs=dvfs_configs,
                                                                kernel_parameters=kernel_parameters.to_dict())
        elif is_energy(metric):
            kernel_metric_data = DataHandler.get_kernel_energy(data_attr, dvfs_configs=dvfs_configs,
                                                               kernel_parameters=kernel_parameters.to_dict())
        else:
            raise Exception("Unknown metric")
    else:
        kernel_metric_data = None

    # We get predictions whether for plotting (3 configs) or reporting table (3K configs)
    label = DataHandler.get_metric_label(metric)
    if KernelsChecks.is_supported(kernel_name):
        cached_prediction_file = runtime_prediction_files_cache.get(kernel_name)
        runtime_prediction = load_cached_prediction(kernel, dvfs_configs, cached_file_path=cached_prediction_file)
        metric_prediction = runtime_prediction
        if is_energy(metric):
            cached_prediction_file = power_prediction_files_cache.get(kernel_name)
            power_prediction = load_cached_prediction(kernel, dvfs_configs, cached_file_path=cached_prediction_file)
            cols = metric_prediction.columns
            runtime_label = DataHandler.get_metric_label(S_RUNTIME)
            power_label = DataHandler.get_metric_label(S_POWER)
            metric_prediction = metric_prediction.merge(power_prediction, how='left',
                                                        on=[i for i in cols if i not in [runtime_label, power_label]])

            metric_prediction[label] = metric_prediction[runtime_label] * metric_prediction[power_label] / 1e3
            metric_prediction.drop(columns=[runtime_label, power_label], inplace=True)
    else:
        # If this is an unsupported kernel, we assume a perfect prediction (measured value)
        if kernel_metric_data is not None:
            metric_prediction = kernel_metric_data.copy()
        else:
            metric_prediction = pd.DataFrame(dvfs_configs)
            metric_prediction[label] = -1

    metric_prediction.rename(columns={label: prediction_key}, inplace=True)

    metric_prediction.drop(columns=[i for i in metric_prediction.columns if i not in
                                    [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, prediction_key]], inplace=True)
    if kernel_metric_data is not None:
        kernel_metric_data.drop(columns=[i for i in kernel_metric_data.columns if i not in
                                         [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, label]], inplace=True)

        result = metric_prediction.merge(kernel_metric_data, how='left', on=[S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ])
    else:
        result = metric_prediction
        result[label] = -1

    result[S_SUPPORTED] = KernelsChecks.is_supported(kernel_name)
    result[S_KERNEL] = kernel_name
    return result


def estimates_versus_metric(nws, node, metric, dvfs_file, supported_only=False, parallel_backend=None):
    """ This method collects the inference time of all networks, and sums up the runtime of all kernels (supported and
    unsupported), and it estimates the networks inference time using the models and summarize the data in a dataframe"""

    dvfs_configs = parse_dvfs_configs(target_file=dvfs_file)

    def internal_method(_nw, _node, _supported_only):
        _data = list()
        # If we didn't characterize a DVFS config, we return a None placeholder instead of exception.
        nw_data_attr = DataAttributes(node=_node, category=S_CHARACTERIZATION, benchmark=_nw, return_placeholder=True,
                                      return_all_columns=True)

        if is_runtime(metric):
            nw_inference_metric_data = DataHandler.get_nw_runtime(nw_data_attr, dvfs_configs)
            nw_inference_metric_key = S_NW_INFERENCE_TIME
            nw_inference_prediction_metric_key = S_PREDICTED_RUNTIME
            kernel_metric_key = S_RUNTIME_MS
            supported_kernels_metric_key = S_SUM_OF_SUPPORTED_KERNELS_RUNTIME
            all_kernels_metric_key = S_SUM_OF_ALL_KERNELS_RUNTIME
        elif is_energy(metric):
            nw_inference_metric_data = DataHandler.get_nw_energy(nw_data_attr, dvfs_configs)
            nw_inference_metric_key = S_NW_INFERENCE_ENERGY
            nw_inference_prediction_metric_key = S_PREDICTED_ENERGY
            kernel_metric_key = S_ENERGY_J
            supported_kernels_metric_key = S_SUM_OF_SUPPORTED_KERNELS_ENERGY
            all_kernels_metric_key = S_SUM_OF_ALL_KERNELS_ENERGY
        else:
            raise Exception(f"Unsupported metric. {metric}")

        if nw_inference_metric_data is None:
            _data.append({S_NETWORK: _nw, S_DVFS_CONFIG: None, nw_inference_metric_key: None,
                          supported_kernels_metric_key: None, all_kernels_metric_key: None,
                          nw_inference_prediction_metric_key: None})
        else:
            # All kernels
            kernels = RuntimeAnalyzer.get_nw_kernels(_nw, _supported_only)
            tmp = list()
            for kernel in kernels:
                x = get_kernel_metric_data(_nw, _node, kernel, metric, dvfs_configs)
                tmp.append(x)
            nw_kernels_metric_data = pd.concat(tmp)

            nw_kernels_metric_data[S_NETWORK] = _nw
            nw_kernels_metric_data[S_DVFS_CONFIG] = None
            frequency_cols = [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ]
            nw_kernels_metric_data[S_DVFS_CONFIG] = \
                nw_kernels_metric_data[frequency_cols].apply(lambda row: DVFSHelpers.get_dvfs_config_level(row), axis=1)

            nw_inference_metric_data[S_DVFS_CONFIG] = \
                nw_inference_metric_data[frequency_cols].apply(lambda row: DVFSHelpers.get_dvfs_config_level(row),
                                                               axis=1)

            nw_inference_metric_data.drop(columns=frequency_cols, inplace=True)
            nw_kernels_metric_data.drop(columns=frequency_cols, inplace=True)
            for dvfs_config_lvl, nw_kernel_grouped_data in nw_kernels_metric_data.groupby(by=S_DVFS_CONFIG):
                nw_inference_metric_value = \
                    float(nw_inference_metric_data[nw_inference_metric_data[S_DVFS_CONFIG] == dvfs_config_lvl]
                          [kernel_metric_key])

                _data.append({S_NETWORK: _nw,
                              S_DVFS_CONFIG: dvfs_config_lvl,
                              nw_inference_metric_key: nw_inference_metric_value,
                              supported_kernels_metric_key: nw_kernel_grouped_data[nw_kernel_grouped_data[S_SUPPORTED]]
                              [kernel_metric_key].sum(),
                              all_kernels_metric_key: nw_kernel_grouped_data[kernel_metric_key].sum(),
                              nw_inference_prediction_metric_key:
                                  nw_kernel_grouped_data[nw_kernel_grouped_data[S_SUPPORTED]]
                                  [nw_inference_prediction_metric_key].sum()})
        return _data

    results = \
        parallel_backend(delayed(internal_method)(nw, node, supported_only) for nw in nws)
    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results)
    return df


def cache_predictions_of_kernel(target_kernel, nws, metric, dvfs_configs, selection_filter, overwrite=False):
    """ Use as a standalone method that gets all kernels from all networks and cache their prediction results. """
    label = DataHandler.get_metric_label(metric)
    seen_before = set()
    filtered_kernels = list()
    for nw in nws:
        kernels = RuntimeAnalyzer.get_nw_kernels(nw, supported_only=True)
        target_kernel = KernelsChecks.get_unified_kernel_name(target_kernel)
        for kernel in kernels:
            kernel_name = KernelsChecks.get_unified_kernel_name(list(kernel.keys())[0])
            if kernel_name != target_kernel or KernelsChecks.is_unsupported(kernel_name):
                continue
            # Remove redundant configs
            kernel_parameters = list(kernel.values())[0]
            kernel_parameters.to_csv()
            key = str(kernel_parameters.to_csv(delimiter='_'))
            if seen_before.__contains__(key):
                continue
            filtered_kernels.append(kernel)
            seen_before.add(key)

    logger.info(f"Number of unique kernel configurations: {len(filtered_kernels)}")

    model_parent_dir = ModelLoader.get_trained_model_dir(target_kernel, label=label, selection_filter=selection_filter)
    cache_predictions(filtered_kernels, dvfs_configs, metric=metric, model_dir=model_parent_dir, overwrite=overwrite)


class NetworkPredictions:
    saved_data_file_for_tables = join_path(NETWORKS_PREDICTION_DIR, 'networks_prediction_for_tables.csv')
    saved_data_file_for_plots = join_path(NETWORKS_PREDICTION_DIR, 'networks_prediction_for_plots.csv')

    @staticmethod
    def add_prediction_errors(df: pd.DataFrame) -> pd.DataFrame:
        # Add runtime prediction error
        df[S_NW_INFERENCE_RUNTIME_PREDICTION_ERR] = \
            Metrics.relative_error_percentage(df[S_NW_INFERENCE_TIME], df[S_PREDICTED_RUNTIME], absolute=False)

        # Add energy prediction error
        df[S_NW_INFERENCE_ENERGY_PREDICTION_ERR] = \
            Metrics.relative_error_percentage(df[S_NW_INFERENCE_ENERGY], df[S_PREDICTED_ENERGY], absolute=False)

        return df

    @staticmethod
    def add_baselines_errors(df: pd.DataFrame) -> pd.DataFrame:
        # Add runtime prediction error
        for i in [S_SUM_OF_SUPPORTED_KERNELS_RUNTIME, S_SUM_OF_ALL_KERNELS_RUNTIME]:
            df[f'{S_NW_INFERENCE_TIME} vs {i} Relative Error %'] = \
                Metrics.relative_error_percentage(df[S_NW_INFERENCE_TIME], df[i], absolute=False)

        # Add energy prediction error
        for i in [S_SUM_OF_SUPPORTED_KERNELS_ENERGY, S_SUM_OF_ALL_KERNELS_ENERGY]:
            df[f'{S_NW_INFERENCE_ENERGY} vs {i} Relative Error %'] = \
                Metrics.relative_error_percentage(df[S_NW_INFERENCE_ENERGY], df[i], absolute=False)

        return df

    @staticmethod
    def concat_predictions(runtime_df: pd.DataFrame, energy_df: pd.DataFrame):
        return pd.merge(runtime_df, energy_df, on=[S_NETWORK, S_DVFS_CONFIG, S_NODE_ID])

    @staticmethod
    def run_predictions(nws, parallel, dvfs_file):
        nodes = ['node-02', 'node-13']

        aggregate_runtime_df = pd.DataFrame()
        aggregate_energy_df = pd.DataFrame()

        for node in nodes:
            runtime_df = estimates_versus_metric(nws=nws, node=node, metric=S_RUNTIME, dvfs_file=dvfs_file,
                                                 parallel_backend=parallel)
            runtime_df[S_NODE_ID] = node
            aggregate_runtime_df = aggregate_runtime_df.append(runtime_df)

            energy_df = estimates_versus_metric(nws=nws, node=node, metric=S_ENERGY, dvfs_file=dvfs_file,
                                                parallel_backend=parallel)
            energy_df[S_NODE_ID] = node
            aggregate_energy_df = aggregate_energy_df.append(energy_df)

        return aggregate_runtime_df, aggregate_energy_df

    @staticmethod
    def print_report(overwrite_data=False):

        # Load saved data file, or create it
        df = NetworkPredictions.main(overwrite=overwrite_data, overwrite_cache=False, dvfs_file=CACHING_DVFS_FILE)

        keep_columns = [S_NETWORK, S_NW_INFERENCE_TIME, S_PREDICTED_RUNTIME, S_NW_INFERENCE_ENERGY, S_PREDICTED_ENERGY]
        df.drop(columns=[i for i in df.columns if i not in keep_columns], inplace=True)

        runtime_df = df[[S_NETWORK, S_NW_INFERENCE_TIME, S_PREDICTED_RUNTIME]].copy()
        energy_df = df[[S_NETWORK, S_NW_INFERENCE_ENERGY, S_PREDICTED_ENERGY]].copy()

        print(f"Average runtime relative error: "
              f"{Metrics.relative_error_percentage(runtime_df[S_NW_INFERENCE_TIME], runtime_df[S_PREDICTED_RUNTIME]).mean()}")
        print(f"Average energy relative error: "
              f"{Metrics.relative_error_percentage(energy_df[S_NW_INFERENCE_ENERGY], energy_df[S_PREDICTED_ENERGY]).mean()}")

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

        # Format energy data
        energy_data = list()
        for name, nw_data in energy_df.groupby(by=[S_NETWORK]):
            results = Metrics.combined_prediction_evaluation_metrics(nw_data[S_NW_INFERENCE_ENERGY],
                                                                     nw_data[S_PREDICTED_ENERGY], as_type='pretty_dict')
            tmp = dict()
            tmp[P_NETWORK] = get_print_benchmark_name(name)
            tmp.update(results)
            tmp['RMSE (W)'] = tmp.pop('RMSE')
            tmp['MAE (W)'] = tmp.pop('MAE')
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
        energy_df = energy_df.round(2)

        # Print both tables
        print("Runtime")
        latex_code = to_latex(runtime_df)
        print(latex_code)

        print("Energy")
        latex_code = to_latex(energy_df)
        print(latex_code)

    @staticmethod
    def main(overwrite=False, overwrite_cache=False, dvfs_file=PLOTTING_DVFS_FILE):
        if dvfs_file is None:
            logger.error("Specify DVFS file. Exiting.")
            return
        if dvfs_file == CACHING_DVFS_FILE:
            NetworkPredictions.saved_data_file = NetworkPredictions.saved_data_file_for_tables
        else:
            NetworkPredictions.saved_data_file = NetworkPredictions.saved_data_file_for_plots

        if os.path.exists(NetworkPredictions.saved_data_file) and not overwrite:
            return pd.read_csv(NetworkPredictions.saved_data_file)

        # TODO modify if needed.
        runtime_models_filters = ModelLoader.get_runtime_models_timestamp()
        power_models_filters = ModelLoader.get_power_models_timestamp()

        ModelLoader.load_specific_models(S_RUNTIME_MS, runtime_prediction_models_cache, runtime_models_filters)
        ModelLoader.load_specific_models(S_AVG_PWR_W, power_prediction_models_cache, power_models_filters)

        for kernel in KernelsChecks.get_supported_kernels():
            runtime_prediction_files_cache[kernel] = \
                join_path(ModelLoader.get_trained_model_dir(kernel, S_RUNTIME_MS, runtime_models_filters.get(kernel)),
                          f'{kernel}_cached_predictions.csv')
            power_prediction_files_cache[kernel] = \
                join_path(ModelLoader.get_trained_model_dir(kernel, S_AVG_PWR_W, power_models_filters.get(kernel)),
                          f'{kernel}_cached_predictions.csv')
        nws = NETWORKS
        nws.sort()
        nws = nws
        n_jobs = N_CPU_CORES if HOSTNAME != 'xps' else 1
        with Parallel(n_jobs=min(len(nws), n_jobs)) as parallel:
            if overwrite_cache:
                load_all_caches(runtime_models_filters, power_models_filters, parallel)
                logger.info("Finished loading all caches.")
            runtime_df, energy_df = NetworkPredictions.run_predictions(nws, parallel, dvfs_file=dvfs_file)

        concat = NetworkPredictions.concat_predictions(runtime_df, energy_df)
        with_errors = NetworkPredictions.add_prediction_errors(concat)
        with_errors.to_csv(NetworkPredictions.saved_data_file, index=False)
        return with_errors


class NetworkPredictionsPlots:
    @staticmethod
    def set_global_fonts():
        # Set figures fonts
        small = 12
        medium = 18
        big = 20
        PlottingHelpers.set_fonts(small, medium, big)

    @staticmethod
    def plot_relative_error(df: pd.DataFrame, figure_path='networks_relative_error.png'):
        # We remove the lowest frequency because of the high errors due to CPU vs GPU activity timing.
        df.drop(inplace=True, index=df[df[S_DVFS_CONFIG] == 'low'].index)
        # --------------------------------------------------------------------------------------------------------------
        # Add the two baselines
        # --------------------------------------------------------------------------------------------------------------
        df = NetworkPredictions.add_baselines_errors(df)
        # **************************************************************************************************************

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
                    f"{runtime_df[runtime_df['Baseline'] == 'Inference Runtime Prediction']['Error %'].abs().mean()}")
        logger.info(f"Average energy absolute relative error: "
                    f"{energy_df[energy_df['Baseline'] == 'Inference Energy Prediction']['Error %'].abs().mean()}")

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
        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=900)
        # **************************************************************************************************************

    @staticmethod
    def main(overwrite_data=False, overwrite_figures=False):
        if overwrite_data:
            DataHandler.clear_cached_summary(category=S_CHARACTERIZATION)

        # Load saved data file, or create it
        if not os.path.exists(NetworkPredictions.saved_data_file_for_plots) or overwrite_data:
            NetworkPredictions.main(overwrite=overwrite_data, overwrite_cache=False, dvfs_file=PLOTTING_DVFS_FILE)

        figure_name = 'networks_relative_error.png'
        os.makedirs(PREDICTION_FIGURES_DIR, exist_ok=True)
        figure_path = join_path(PREDICTION_FIGURES_DIR, figure_name)
        if not os.path.exists(figure_path) or overwrite_figures:
            df = pd.read_csv(NetworkPredictions.saved_data_file_for_plots)
            NetworkPredictionsPlots.set_global_fonts()
            NetworkPredictionsPlots.plot_relative_error(df, figure_path)


def load_all_caches(runtime_selection_filter, power_selection_filter, parallel_backend):
    dvfs_configs = parse_dvfs_configs(CACHING_DVFS_FILE)

    def _load_all_caches(kernel, _runtime_selection_filters, _power_selection_filters):
        _runtime_selection_filter = _runtime_selection_filters.get(kernel)
        _power_selection_filter = _power_selection_filters.get(kernel)

        runtime_prediction_files_cache[kernel] = \
            join_path(ModelLoader.get_trained_model_dir(kernel, S_RUNTIME_MS, _runtime_selection_filter),
                      f'{kernel}_cached_predictions.csv')
        power_prediction_files_cache[kernel] = \
            join_path(ModelLoader.get_trained_model_dir(kernel, S_AVG_PWR_W, _power_selection_filter),
                      f'{kernel}_cached_predictions.csv')

        FileUtils.silent_remove(runtime_prediction_files_cache[kernel])
        FileUtils.silent_remove(power_prediction_files_cache[kernel])

        cache_predictions_of_kernel(target_kernel=kernel, nws=NETWORKS, metric=S_RUNTIME,
                                    dvfs_configs=dvfs_configs,
                                    selection_filter=_runtime_selection_filter)
        cache_predictions_of_kernel(target_kernel=kernel, nws=NETWORKS, metric=S_POWER,
                                    dvfs_configs=dvfs_configs,
                                    selection_filter=_power_selection_filter)

    parallel_backend(delayed(_load_all_caches)(kernel, runtime_selection_filter, power_selection_filter)
                     for kernel in KernelsChecks.get_supported_kernels())


def test():
    dvfs_configs = parse_dvfs_configs(CACHING_DVFS_FILE)
    selection_filter = ModelSelectionFilter(n_from_last=3)
    runtime_prediction_models_cache['conv2d'] = TrainedModelWrapper(
        ModelLoader.load_trained_model('conv2d', S_RUNTIME_MS,
                                       selection_filter))
    cache_predictions_of_kernel(target_kernel='conv2d', nws=NETWORKS, metric=S_RUNTIME,
                                dvfs_configs=dvfs_configs,
                                selection_filter=selection_filter)


if __name__ == '__main__':
    # For comprehensive results
    NetworkPredictions.print_report(overwrite_data=True)

    # For plotting
    # NetworkPredictions.main(overwrite=True, overwrite_cache=False, dvfs_file=PLOTTING_DVFS_FILE)
    # NetworkPredictionsPlots.main(overwrite_data=True, overwrite_figures=True)
