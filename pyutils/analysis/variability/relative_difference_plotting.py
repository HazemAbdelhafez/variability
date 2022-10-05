import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pyutils.analysis.variability.strings import S_INTRA, S_INTER, S_VAR_CATEGORY, S_PERCENTAGE_DIFF
from pyutils.characterization.networks.utils import get_print_benchmark_name
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_NODE_ID, S_BENCHMARK
from pyutils.common.utils import TimeStamp, prepare, get_figures_dir_path, GlobalLogger
from pyutils.run.analysis.config import S_TIMESTAMP, S_PLOTTING, S_BENCHMARKS, S_METRICS, S_FONT, \
    S_FIG_SIZE, \
    S_SHOWFLIERS, \
    S_WHIS, S_FIGURE_NAME


logger = GlobalLogger().get_logger()


class BenchmarkAnalysisPlots:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ts_obj = TimeStamp.parse_timestamp(cfg[S_TIMESTAMP])
        self.figures_path = prepare(get_figures_dir_path(__file__), self.ts_obj)
        self.plotting_cfg = cfg[S_PLOTTING]
        self.benchmarks: list = self.cfg[
            S_BENCHMARKS]  # In case we needed to plot a sub-set of the benchmarks
        self.dims = [len(self.benchmarks), len(self.cfg[S_METRICS])]
        PlottingHelpers.init()
        PlottingHelpers.set_font(self.plotting_cfg[S_FONT])
        self.axs = dict()
        self.figs = dict()

    def get_fig_and_ax(self, variability):
        if variability not in self.axs.keys():
            # In case we specify a subset after initializing the class instance.
            self.dims = [len(self.benchmarks), len(self.cfg[S_METRICS])]
            rows, cols = self.dims

            fig_size = [self.plotting_cfg[S_FIG_SIZE][0] * cols, self.plotting_cfg[S_FIG_SIZE][1] * rows]
            # Initialize subplots
            fig, axs = plt.subplots(rows, cols, figsize=fig_size)
            axs = np.reshape(axs, (rows, cols))
            self.axs[variability] = axs
            self.figs[variability] = fig
        return self.figs[variability], self.axs[variability]

    def medians_difference(self, bm, metric, df: pd.DataFrame, prefix=None):
        """ Plots the medians difference (i.e., intra- and inter-node variability indicators) together
        for several metrics (e.g., runtime, power, etc.). """

        i = self.benchmarks.index(bm)
        j = self.cfg[S_METRICS].index(metric)
        fig, axs = self.get_fig_and_ax("combined" if prefix is None else f"combined_{prefix}")
        ax = axs[i][j]
        df = df.drop(index=df[df[S_INTER] == np.NaN].index)
        df = df.drop(index=df[df[S_INTRA] == np.NaN].index)
        logger.info(f"Benchmark: {bm}, Metric: {metric}, Shape: {df.shape}")
        whis = self.cfg[S_PLOTTING][S_WHIS]
        show_fliers = self.cfg[S_PLOTTING][S_SHOWFLIERS]
        data = pd.melt(df, id_vars=[S_NODE_ID, S_BENCHMARK], value_vars=[S_INTRA, S_INTER],
                       var_name=S_VAR_CATEGORY, value_name=S_PERCENTAGE_DIFF)

        sns.boxplot(x=S_NODE_ID, y=S_PERCENTAGE_DIFF, hue=S_VAR_CATEGORY, data=data, showfliers=show_fliers,
                    whis=whis, ax=ax)
        sns.stripplot(x=S_NODE_ID, y=S_PERCENTAGE_DIFF, hue=S_VAR_CATEGORY, color='.3', data=data,
                      jitter=0.08,
                      size=0.6, dodge=True, ax=ax)

        # Set box-plots to gray scale
        PlottingHelpers.remove_y_axis_label(ax)
        if j != 0:
            PlottingHelpers.remove_y_axis_label(ax)
        if i == 0:
            if j == 0:
                ax.set_title(metric, weight='bold')
            if j == 1:
                ax.set_title(metric, weight='bold')
        if i == self.dims[0] - 1:
            PlottingHelpers.set_xaxis_label(ax, 'Node ID')
        else:
            PlottingHelpers.remove_x_axis_labels_and_ticks(ax)

        # Set box-plots to gray scale
        new_handles = PlottingHelpers.set_to_grey_scale(ax, hue_data=data[S_VAR_CATEGORY].unique())

        PlottingHelpers.remove_legend(ax)

        if i == 0:
            num_unique_labels = len(data[S_VAR_CATEGORY].unique())
            _, legend_labels = ax.get_legend_handles_labels()[:num_unique_labels]
            legend_labels = [str(item).capitalize() for item in legend_labels[:num_unique_labels]]
            ax.legend(new_handles, legend_labels, bbox_to_anchor=(1, 1.15), loc='upper right',
                      ncol=len(data[S_VAR_CATEGORY].unique()), frameon=False, handletextpad=0.3,
                      borderpad=0.2, labelspacing=0.1, handlelength=0.8)

        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.text(x=0.01, y=1.04, transform=ax.transAxes, s=get_print_benchmark_name(bm), style='italic')
        fig.text(0.005, 0.5, 'Relative difference %', ha='center', va='center', rotation='vertical')

        if prefix is not None:
            fig_name = f"{prefix}_combined_{self.cfg[S_FIGURE_NAME]}"
        else:
            fig_name = f"combined_{self.cfg[S_FIGURE_NAME]}"

        fig_path = prepare(self.figures_path, "combined", fig_name)
        fig.savefig(fig_path, transparent=False, bbox_inches='tight', pad_inches=0.03, dpi=300)
        logger.info(f"Saving figure to: {fig_path}")
        plt.close(fig)

    def generic_node_median_difference(self, bm, metric, variability, df: pd.DataFrame, prefix=None):
        """ Plots the medians difference (e.g., intra-node variability indicators) for multiple benchmarks
                on multiple subplots for several metrics (e.g., runtime, power, etc.), for a specified
                variability category. """

        i = self.benchmarks.index(bm)
        j = self.cfg[S_METRICS].index(metric)
        fig, axs = self.get_fig_and_ax(variability if prefix is None else f"{variability}_{prefix}")
        ax = axs[i][j]
        df = df.drop(index=df[df[variability] == np.NaN].index)
        logger.info(f"Benchmark: {bm}, Metric: {metric}, Variability: {variability}, Shape: {df.shape}")

        whis = self.cfg[S_PLOTTING][S_WHIS]
        show_fliers = self.cfg[S_PLOTTING][S_SHOWFLIERS]
        sns.stripplot(x=S_NODE_ID, y=variability, data=df, jitter=0.2, size=1.5, color=".3", ax=ax)
        sns.boxplot(x=S_NODE_ID, y=variability, data=df, showfliers=show_fliers, whis=whis, ax=ax)

        # Set box-plots to gray scale
        PlottingHelpers.set_to_grey_scale(ax)
        PlottingHelpers.remove_y_axis_label(ax)

        ax.text(x=0.01, y=1.04, transform=ax.transAxes, s=get_print_benchmark_name(bm), style='italic')
        PlottingHelpers.remove_legend(ax)

        if j != 0:
            PlottingHelpers.remove_y_axis_label(ax)

        if i == 0:
            if j == 0:
                ax.set_title('Power', weight='bold')
            if j == 1:
                ax.set_title('Runtime', weight='bold')

        if i == self.dims[0] - 1:
            PlottingHelpers.set_xaxis_label(ax, 'Node ID')
        else:
            PlottingHelpers.remove_x_axis_labels_and_ticks(ax)

        ax.yaxis.grid(True)

        fig.text(0.01, 0.5, 'Relative difference %', ha='center', va='center', rotation='vertical')

        if prefix is not None:
            fig_name = f"{prefix}_{variability}_{self.cfg[S_FIGURE_NAME]}"
        else:
            fig_name = f"{variability}_{self.cfg[S_FIGURE_NAME]}"

        fig_path = prepare(self.figures_path, variability, fig_name)
        fig.savefig(fig_path, transparent=False, bbox_inches='tight', pad_inches=0.03, dpi=300)
        plt.close(fig)

    def intra_node_median_difference(self, bm, metric, df: pd.DataFrame, prefix=None):
        """ Plots the medians difference (i.e., intra-node variability indicators) for multiple benchmarks
        on multiple subplots for several metrics (e.g., runtime, power, etc.). """
        self.generic_node_median_difference(bm, metric, S_INTRA, df, prefix)

    def inter_node_median_difference(self, bm, metric, df: pd.DataFrame, prefix=None):
        """ Plots the medians difference (i.e., inter-node variability indicators) for multiple benchmarks
        on multiple subplots for several metrics (e.g., runtime, power, etc.). """
        self.generic_node_median_difference(bm, metric, S_INTER, df, prefix)
