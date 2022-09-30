import ast
import os.path
import platform
from os.path import join as jp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

import pyutils.hosts.agx as agx
from pyutils.analysis.common.data_transformers import Transformers
from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.common.config import ALL_LABELS
from pyutils.common.methods import is_runtime, is_power
from pyutils.common.paths import MODELS_PERFORMANCE_DIR, MODELS_PERFORMANCE_FIGURES_DIR
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_RUNTIME_MS, S_METRIC, S_RUNTIME, S_POWER, S_AVG_PWR_W, S_PREDICTED_RUNTIME, \
    S_KERNEL
from pyutils.common.timers import StopWatch
from pyutils.common.utils import GlobalLogger
from pyutils.hosts.agx import DVFSHelpers
from pyutils.modeling.helpers import ModelLoader, TrainingHelpers

ALL_KERNELS = KernelsChecks.get_supported_kernels()
# TIMING_ITERATIONS = 100
TIMING_ITERATIONS = 200
WARMUP_ITERATIONS = 10

S_BATCH_SIZE = "Batch Size"
# L_NUMBER_OF_CONFIGURATIONS = [1] + list(range(500, 10500, 500))
L_NUMBER_OF_CONFIGURATIONS = [1] + list(range(5, 105, 5))
MIN_BATCH_SIZE = min(L_NUMBER_OF_CONFIGURATIONS)
MAX_BATCH_SIZE = max(L_NUMBER_OF_CONFIGURATIONS)
AVG_BATCH_SIZE = int(np.mean(L_NUMBER_OF_CONFIGURATIONS))

NUMBER_OF_CONFIGURATIONS = 1000
S_PREDICTION_TIME = 'prediction_time'

logger = GlobalLogger().get_logger()


class PredictionOverhead:
    @staticmethod
    def evaluate_kernel_prediction_performance(kernel_name, prediction_model):
        kernel_training_model = TrainingHelpers.load_kernel_model(kernel_name)
        kernel_parameters_generator = KernelsLoaders.load_generator(kernel_name)
        kernel_parameters = kernel_parameters_generator.generate_random_input_parameters()
        kernel_parameters = kernel_training_model.preprocess_kernel_parameters(kernel_parameters)

        # Prepare prediction input
        dvfs_config = DVFSHelpers.generate_random_dvfs_config()
        prediction_input = pd.DataFrame({**kernel_parameters, **dvfs_config}, index=[0])
        prediction_value = None

        # Runtime prediction performance
        timer = StopWatch()
        prediction_input.drop(columns=[i for i in prediction_input.columns if i in ALL_LABELS], inplace=True)
        prediction_input_dm = xgb.DMatrix(prediction_input)
        prediction_model.model.set_param({'nthread': 6})

        warmup_list = list()
        for _ in range(WARMUP_ITERATIONS):
            timer.start()
            prediction_value = prediction_model.model.predict(prediction_input_dm)
            prediction_value = Transformers.inverse_log_transform(prediction_value)[0]
            warmup_list.append(timer.elapsed_ms())

        runtime = list()
        for _ in range(TIMING_ITERATIONS + WARMUP_ITERATIONS):
            timer.start()
            prediction_value = prediction_model.model.predict(prediction_input_dm)
            prediction_value = Transformers.inverse_log_transform(prediction_value)[0]
            runtime.append(timer.elapsed_ms())

        mean_runtime = np.mean(runtime[WARMUP_ITERATIONS:])
        return mean_runtime, prediction_value

    @staticmethod
    def evaluate_inference_performance(metric=S_RUNTIME, overwrite=False):
        output_file_path = \
            jp(MODELS_PERFORMANCE_DIR,
               f'kernels_{metric}_{len(ALL_KERNELS)}_{NUMBER_OF_CONFIGURATIONS}_{TIMING_ITERATIONS}.csv')
        if os.path.exists(output_file_path) and not overwrite:
            logger.warning(f"Inference performance file exists, and not overwritten enabled. Exiting.")
            return pd.read_csv(output_file_path)

        if platform.processor() != 'aarch64':
            logger.warning(f"Skip the runtime models performance data collection phase. "
                           f"It must be run on a Jetson board")

        agx.BoardController.reset_dvfs_settings()

        # For each kernel, run for x iterations, generate random configs, and report mean/boxplot
        runtime_selection_filters = ModelLoader.get_runtime_models_timestamp()
        power_selection_filters = ModelLoader.get_power_models_timestamp()

        models_cache = dict()
        if is_runtime(metric):
            ModelLoader.load_specific_models(S_RUNTIME_MS, models_cache, runtime_selection_filters)
        elif is_power(metric):
            ModelLoader.load_specific_models(S_AVG_PWR_W, models_cache, power_selection_filters)
        else:
            logger.error(f"Invalid metric: {metric}. Exiting.")
            return

        result = dict()
        result[S_KERNEL] = list()
        result[S_PREDICTED_RUNTIME] = list()
        result[S_PREDICTION_TIME] = list()
        result[S_METRIC] = list()

        for kernel in ALL_KERNELS:
            for _ in range(NUMBER_OF_CONFIGURATIONS):
                prediction_time, predicted_metric = \
                    PredictionOverhead.evaluate_kernel_prediction_performance(kernel, models_cache.get(kernel))
                result[S_KERNEL].append(kernel)
                result[S_METRIC].append(metric)
                result[S_PREDICTED_RUNTIME].append(predicted_metric)
                result[S_PREDICTION_TIME].append(prediction_time)

        summary = pd.DataFrame(result)

        try:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

            os.makedirs(MODELS_PERFORMANCE_DIR, exist_ok=True)
            summary.to_csv(output_file_path, index=False)
        except Exception as e:
            logger.error("An error occurred, saving to temp file.")
            summary.to_csv(f'{metric}_tmp.csv', index=False)
        finally:
            return summary


class BatchSizeImpact:
    @staticmethod
    def evaluate_kernel_batched_prediction_performance(kernel_name, prediction_model):
        kernel_training_model = TrainingHelpers.load_kernel_model(kernel_name)
        kernel_parameters_generator = KernelsLoaders.load_generator(kernel_name)

        batch = list()
        for _ in range(MAX_BATCH_SIZE):
            kernel_parameters = kernel_parameters_generator.generate_random_input_parameters()
            kernel_parameters = kernel_training_model.preprocess_kernel_parameters(kernel_parameters)

            # Prepare prediction input
            dvfs_config = DVFSHelpers.generate_random_dvfs_config()
            batch.append({**kernel_parameters, **dvfs_config})

        prediction_input = pd.DataFrame(batch)
        prediction_input.drop(columns=[i for i in prediction_input.columns if i in ALL_LABELS], inplace=True)
        prediction_model.model.set_param({'nthread': 6})
        prediction_value = -1  # Placeholder

        results = list()
        warmup_list = [1, 2, 4]
        for batch_size in warmup_list + L_NUMBER_OF_CONFIGURATIONS:
            logger.info(f"At batch size: {batch_size}")
            result = dict()
            # Runtime prediction performance
            timer = StopWatch()
            prediction_input_dm = xgb.DMatrix(prediction_input.head(batch_size))
            runtime = list()
            for _ in range(TIMING_ITERATIONS + WARMUP_ITERATIONS):
                timer.start()
                prediction_value = prediction_model.model.predict(prediction_input_dm)
                prediction_value = Transformers.inverse_log_transform(prediction_value)
                runtime.append(timer.elapsed_ms())

            result[S_KERNEL] = kernel_name
            result[S_BATCH_SIZE] = batch_size
            result[S_RUNTIME_MS] = runtime[WARMUP_ITERATIONS:]
            results.append(result)

        return results[len(warmup_list):], prediction_value

    @staticmethod
    def evaluate_batched_inference_performance(metric=S_RUNTIME, overwrite=False):
        output_file_path = \
            jp(MODELS_PERFORMANCE_DIR,
               f'kernels_{metric}_{MIN_BATCH_SIZE}_{AVG_BATCH_SIZE}_{MAX_BATCH_SIZE}_{TIMING_ITERATIONS}.csv')

        if os.path.exists(output_file_path) and not overwrite:
            logger.warning(f"Inference performance file exists, and not overwritten enabled. Exiting.")
            df = pd.read_csv(output_file_path)
            df[S_RUNTIME_MS] = df.apply(lambda x: np.median(ast.literal_eval(x[S_RUNTIME_MS])), axis=1)
            return df

        if platform.processor() != 'aarch64':
            logger.warning(f"Skip the runtime models performance data collection phase. "
                           f"It must be run on a Jetson board")
        agx.BoardController.reset_dvfs_settings()

        # For each kernel, run for x iterations, generate random configs, and report mean/boxplot
        runtime_selection_filters = ModelLoader.get_runtime_models_timestamp()
        power_selection_filters = ModelLoader.get_power_models_timestamp()

        models_cache = dict()
        if is_runtime(metric):
            ModelLoader.load_specific_models(S_RUNTIME_MS, models_cache, runtime_selection_filters)
        elif is_power(metric):
            ModelLoader.load_specific_models(S_AVG_PWR_W, models_cache, power_selection_filters)
        else:
            logger.error(f"Invalid metric: {metric}. Exiting.")
            return

        combined_stats = list()
        for kernel in ALL_KERNELS:
            stats, _ = BatchSizeImpact.evaluate_kernel_batched_prediction_performance(kernel, models_cache.get(kernel))
            combined_stats += stats
        summary = pd.DataFrame(combined_stats)
        summary[S_METRIC] = metric

        try:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            os.makedirs(MODELS_PERFORMANCE_DIR, exist_ok=True)
            summary.to_csv(output_file_path, index=False)
        except Exception as e:
            logger.error("An error occurred, saving to temp file.")
            summary.to_csv(f'{metric}_tmp.csv', index=False)
        finally:
            return summary


class ModelsPerformance:
    @staticmethod
    def per_record(overwrite_data=False):
        x = PredictionOverhead.evaluate_inference_performance(metric=S_RUNTIME, overwrite=overwrite_data)
        y = PredictionOverhead.evaluate_inference_performance(metric=S_POWER, overwrite=overwrite_data)
        return x, y

    @staticmethod
    def batched(overwrite_data=False):
        x = BatchSizeImpact.evaluate_batched_inference_performance(metric=S_RUNTIME, overwrite=overwrite_data)
        if S_METRIC not in x.columns:
            x[S_METRIC] = S_RUNTIME_MS
        y = BatchSizeImpact.evaluate_batched_inference_performance(metric=S_POWER, overwrite=overwrite_data)
        if S_METRIC not in y.columns:
            y[S_METRIC] = S_POWER
        return x, y


class ModelsPerformancePlot:

    @staticmethod
    def plot_stacked_overhead(runtime_df: pd.DataFrame, power_df: pd.DataFrame, figure_path):
        runtime_df['Overhead %'] = 100 * runtime_df[S_PREDICTION_TIME] / runtime_df[S_PREDICTED_RUNTIME]
        power_df['Overhead %'] = 100 * power_df[S_PREDICTION_TIME] / power_df[S_PREDICTED_RUNTIME]

        df = pd.DataFrame(pd.concat([runtime_df, power_df]))

        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [7, 1]}, figsize=(8, 3.5))

        sns.boxplot(y='Overhead %', x=S_KERNEL, data=df[df[S_KERNEL] == 'add'], hue=S_METRIC, showfliers=False,
                    ax=axs[1])
        sns.boxplot(y='Overhead %', x=S_KERNEL, data=df[df[S_KERNEL] != 'add'], hue=S_METRIC, showfliers=False,
                    ax=axs[0])
        for i in range(len(axs)):
            ax = axs[i]
            x_labels = [str(item.get_text()).capitalize() for item in ax.get_xticklabels()]
            ax.set_xticklabels(x_labels)

            PlottingHelpers.capitalize_x_axis_labels(ax)
            PlottingHelpers.rotate_x_labels(ax, angle=20)
            new_handles = PlottingHelpers.set_to_grey_scale(ax, hue_data=df[S_METRIC].unique())

            _, legend_labels = ax.get_legend_handles_labels()
            legend_labels = [str(item).capitalize() for item in legend_labels]
            ax.legend(new_handles, legend_labels, bbox_to_anchor=(0.55, 1.07), loc='center', ncol=3, frameon=False,
                      handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)

            # Set grid lines for all axes
            ax.yaxis.grid(True)

            # Remove the default SNS xlabel
            ax.setup(xlabel=None)

            if i == 1:
                # Move the Y-axis of the right subplot to the right
                PlottingHelpers.move_y_axis_to_right(ax)

                # Remove the right subplot legend and y-axis label
                PlottingHelpers.remove_legend(ax)

                PlottingHelpers.remove_y_axis_label(ax)

        # Set plots whitespace sizes
        plt.subplots_adjust(wspace=0.001, hspace=0.01)
        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=600)

    @staticmethod
    def plot_prediction_times(runtime_df: pd.DataFrame, power_df: pd.DataFrame, figure_path):
        # TODO: we skip the power plot due to space limitations in the submission. Fix later.
        df = runtime_df

        # TODO: pick worst 4 kernels
        df = df[df[S_KERNEL].isin(['conv2d', 'add', 'cat', 'adaptivepool2d'])].copy()

        df[S_KERNEL] = df[S_KERNEL].apply(lambda x: x.capitalize())

        df.drop(columns=[S_METRIC], inplace=True)
        column_name = "Normalize \nPrediction Time (ms)"
        df.rename(columns={S_RUNTIME_MS: column_name}, inplace=True)
        df[column_name] = df[column_name] / df[S_BATCH_SIZE]

        # Print reduction ratio
        for g, g_d in df.groupby([S_KERNEL]):
            print(g, max(g_d[column_name]) / min(g_d[column_name]))

        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

        sns.lineplot(data=df, y=column_name, x=S_BATCH_SIZE, hue=S_KERNEL, ax=ax, palette='gray')
        ax.set_yscale('log')
        ax.grid(True)

        plt.subplots_adjust(wspace=0.001, hspace=0.01)
        plt.legend(bbox_to_anchor=(0.5, 1.06), loc='center', ncol=5, frameon=False,
                   handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)

        plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=900)

    @staticmethod
    def set_global_fonts():
        # Set figures fonts
        small = 12
        medium = 16
        big = 18
        PlottingHelpers.set_fonts(small, medium, big)

    @staticmethod
    def main_per_record(overwrite_data=False, overwrite_figures=False):
        runtime_models_performance_df, power_models_performance_df = ModelsPerformance.per_record(overwrite_data)

        figure_name = 'models_prediction_performance.png'
        if not os.path.exists(MODELS_PERFORMANCE_FIGURES_DIR):
            logger.warning(f"Figures directory did not exist. Creating it at: {MODELS_PERFORMANCE_FIGURES_DIR}")
            os.makedirs(MODELS_PERFORMANCE_FIGURES_DIR)

        figure_path = jp(MODELS_PERFORMANCE_FIGURES_DIR, figure_name)
        if not os.path.exists(figure_path) or overwrite_figures:
            ModelsPerformancePlot.set_global_fonts()
            ModelsPerformancePlot.plot_stacked_overhead(runtime_models_performance_df, power_models_performance_df,
                                                        figure_path)

    @staticmethod
    def main_batched(overwrite_data=False, overwrite_figures=False):
        runtime_models_performance_df, power_models_performance_df = ModelsPerformance.batched(overwrite_data)

        figure_name = 'models_batched_prediction_performance.png'
        os.makedirs(MODELS_PERFORMANCE_FIGURES_DIR, exist_ok=True)

        figure_path = jp(MODELS_PERFORMANCE_FIGURES_DIR, figure_name)
        if not os.path.exists(figure_path) or overwrite_figures:
            ModelsPerformancePlot.set_global_fonts()
            ModelsPerformancePlot.plot_prediction_times(runtime_models_performance_df, power_models_performance_df,
                                                        figure_path)


if __name__ == '__main__':
    # ModelsPerformance.per_record(overwrite_data=True)
    # ModelsPerformance.batched(overwrite_data=True)
    # ModelsPerformancePlot.main(overwrite_data=False, overwrite_figures=True)
    ModelsPerformancePlot.main_batched(overwrite_data=False, overwrite_figures=True)
