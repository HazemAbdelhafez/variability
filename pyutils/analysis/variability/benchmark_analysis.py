from multiprocessing import Process

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pyutils.analysis.variability.relative_difference_analysis import BenchmarkAnalysis
from pyutils.analysis.variability.relative_difference_plotting import BenchmarkAnalysisPlots
from pyutils.analysis.variability.statistical_analysis.effect_size import ROBUST_D
from pyutils.analysis.variability.statistical_analysis.strings import BONFERRONI_CORRECTED_KS_PVALUE, \
    BH_CORRECTED_KS_PVALUE, AD_k_STATISTIC
from pyutils.characterization.networks.utils import get_print_benchmark_name
from pyutils.common.config import S_CPU_FREQ, \
    S_GPU_FREQ, S_MEMORY_FREQ
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_BENCHMARK, S_METRIC, S_POWER, S_RUNTIME
from pyutils.common.utils import prepare, TimeStamp, get_figures_dir_path, GlobalLogger
from pyutils.run.analysis.config import S_BENCHMARKS, S_PLOTTING, S_FONT, \
    S_METRICS, \
    S_OVERWRITE_STATS, S_K_AD_FIGURE_NAME, S_ROBUST_D_FIGURE_NAME, \
    S_TIMESTAMP, S_INDIVIDUAL_PLOTS, S_WHIS, S_SHOWFLIERS


logger = GlobalLogger().get_logger()


class Driver:
    def benchmarks_analysis(self, benchmark, metric, cfg: dict, bma: BenchmarkAnalysis.__class__):
        logger.info(f"Analyzing {benchmark}:{metric}")
        cfg.update({S_BENCHMARK: benchmark})
        cfg.update({S_METRIC: metric})
        ba = bma(cfg)
        df = ba.calculate_medians_difference()
        df = ba.remove_outliers(df)
        df[S_METRIC] = metric
        return df

    def benchmarks_statistical_analysis(self, cfg: dict, bma: BenchmarkAnalysis.__class__):
        benchmarks = cfg[S_BENCHMARKS]
        metrics = cfg[S_METRICS]

        k_ad_dfs = list()
        two_samples_dfs = list()
        for b in benchmarks:
            cfg.update({S_BENCHMARK: b})
            for m in metrics:
                logger.info(f"Analyzing {b}:{m}")
                cfg.update({S_METRIC: m})
                ba = bma(cfg)
                k_ad_df, ks_df = ba.statistical_analysis(cfg[S_OVERWRITE_STATS])
                k_ad_dfs.append(k_ad_df)
                two_samples_dfs.append(ks_df)

        two_samples_df = pd.concat(two_samples_dfs)
        self.print_ks_p_values(two_samples_df, fdr_correction_algorithm=BH_CORRECTED_KS_PVALUE)
        self.print_ks_p_values(two_samples_df, fdr_correction_algorithm=BONFERRONI_CORRECTED_KS_PVALUE)
        self.plot_k_ad(pd.concat(k_ad_dfs), cfg)
        self.plot_robust_d(two_samples_df, cfg)

    def plot_k_ad(self, df, cfg):
        PlottingHelpers.set_font(cfg[S_PLOTTING][S_FONT])

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        sns.boxplot(x=S_BENCHMARK, y=AD_k_STATISTIC, hue=S_METRIC, data=df, whis=cfg[S_PLOTTING][S_WHIS],
                    showfliers=cfg[S_PLOTTING][S_SHOWFLIERS], ax=ax)
        sns.stripplot(x=S_BENCHMARK, y=AD_k_STATISTIC, hue=S_METRIC, color='.3', data=df, jitter=0.08,
                      size=0.6,
                      dodge=True, ax=ax)
        new_handles = PlottingHelpers.set_to_grey_scale(ax, hue_data=df[S_METRIC].unique())[:2]

        # For 14 nodes: critical values are [3.20211717,  4.07577445]
        ax.set_ylabel('Statistic Value')

        # TODO: we have multiple critical values because when we drop rows that have RME>required_rme
        # we end up with a situation where for some configs, the number of nodes is not 14.
        critical_values = [float(df['critical_value_at_1%'].unique().min())]
        labels = ['1% Significance level']
        styles = ['dotted']
        colors = ['black']
        for i, l, s, c in zip(critical_values, labels, styles, colors):
            ax.axhline(y=i, label=l, linestyle=s, color=c)

        handles, labels = ax.get_legend_handles_labels()
        new_handles = [handles[0]] + new_handles
        ax.legend(new_handles, labels[:3], loc="best", title="", frameon=True, facecolor='white',
                  framealpha=1)

        x_tick_labels = []
        for i in range(len(ax.get_xticklabels())):
            x_tick_labels.append(get_print_benchmark_name(ax.get_xticklabels()[i].get_text()))
        ax.set_xticklabels(x_tick_labels, rotation=15, fontsize=18)

        PlottingHelpers.remove_x_axis_label(ax)
        ax.grid(axis='y')
        fig.suptitle('')
        plt.tight_layout()

        ts_obj = TimeStamp.parse_timestamp(cfg[S_TIMESTAMP])
        fig_path = prepare(get_figures_dir_path(__file__), ts_obj, cfg[S_K_AD_FIGURE_NAME])
        fig.savefig(fig_path, dpi=600, bbox_inches='tight')

    def plot_robust_d(self, df, cfg):
        # Preamble analysis: for zero-valued pooled std of samples, the difference between
        # means is divided by 0., hence, Dr becomes -inf. Here I print how many records are so.
        df = df.copy()

        logger.info(f"Number of Dr values: {df.shape[0]}")

        for gn, gd in df.groupby(by=[S_METRIC]):
            tmp_1 = gd[gd[ROBUST_D] >= 1].shape[0] + gd[gd[ROBUST_D] == float('-inf')].shape[0] + \
                    gd[gd[ROBUST_D] == float('inf')].shape[0]
            tmp_2 = gd[gd[ROBUST_D] >= 2].shape[0] + gd[gd[ROBUST_D] == float('-inf')].shape[0] + \
                    gd[gd[ROBUST_D] == float('inf')].shape[0]
            logger.info(f"-- {gn} Dr values size: {gd.shape[0]}")
            logger.info(f"-- {gn} : Dr>=1 {100 * tmp_1 / gd.shape[0]} - Dr>=2 {100 * tmp_2 / gd.shape[0]}")

        for gn, gd in df.groupby(by=[S_METRIC]):
            tmp = gd[(gd[ROBUST_D] == float('inf')) | (gd[ROBUST_D] == float('-inf'))]
            logger.info(f"-- Metric: {gn}, Inf values: {tmp.shape[0]} out of {gd.shape[0]}")

        df = df[(df[ROBUST_D] != float('-inf')) & (df[ROBUST_D] != float('inf'))]
        logger.info(f"-- Number of Dr values - after removing inf: {df.shape[0]}")

        PlottingHelpers.set_font(cfg[S_PLOTTING][S_FONT])
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))

        # df[ROBUST_D] = np.log(df[ROBUST_D].abs())

        sns.boxplot(x=S_BENCHMARK, y=ROBUST_D, hue=S_METRIC, data=df, whis=cfg[S_PLOTTING][S_WHIS],
                    showfliers=cfg[S_PLOTTING][S_SHOWFLIERS], ax=ax)
        sns.stripplot(x=S_BENCHMARK, y=ROBUST_D, hue=S_METRIC, color='.3', data=df, jitter=0.08,
                      size=0.6, dodge=True, ax=ax)
        new_handles = PlottingHelpers.set_to_grey_scale(ax, hue_data=df[S_METRIC].unique())[:2]

        ax.set_ylabel('Robust D (Dr)')
        # ax.set_ylabel('Robust D (Dr) - Log Scale')

        _, labels = ax.get_legend_handles_labels()
        ax.legend(new_handles, labels[:2], loc="best", title="", frameon=True, facecolor='white',
                  framealpha=1)

        x_tick_labels = []
        for i in range(len(ax.get_xticklabels())):
            x_tick_labels.append(get_print_benchmark_name(ax.get_xticklabels()[i].get_text()))
        ax.set_xticklabels(x_tick_labels, rotation=15, fontsize=18)

        PlottingHelpers.remove_x_axis_label(ax)
        ax.grid(axis='y')
        fig.suptitle('')
        plt.tight_layout()

        ts_obj = TimeStamp.parse_timestamp(cfg[S_TIMESTAMP])
        fig_path = prepare(get_figures_dir_path(__file__), ts_obj, cfg[S_ROBUST_D_FIGURE_NAME])
        logger.info(f"Saving figure to: {fig_path}")
        fig.savefig(fig_path, dpi=600, bbox_inches='tight')

    def print_ks_p_values(self, original_df, fdr_correction_algorithm=BH_CORRECTED_KS_PVALUE):
        logger.info(f"KS tests results with {fdr_correction_algorithm}.")
        keep = [fdr_correction_algorithm, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_BENCHMARK]

        p_value_ranges = [0.01, 0.05]

        modes = [S_POWER, S_RUNTIME]
        report_once = True

        for mode in modes:
            data_df = original_df[original_df[S_METRIC] == mode].copy()
            data_df.drop(columns=[i for i in data_df.columns if i not in keep], inplace=True)

            ranges = dict()
            s_range = 'Alpha'
            ranges[s_range] = [f'{int(100 * i)}%' for i in p_value_ranges]

            for benchmark, df in data_df.groupby(by=[S_BENCHMARK]):
                benchmark = get_print_benchmark_name(benchmark)
                num_records = df.shape[0]

                ratio = df[df[fdr_correction_algorithm] <= p_value_ranges[0]].shape[0] / num_records
                percentage = 100 * ratio
                ranges[benchmark] = [percentage]

                ratio = df[df[fdr_correction_algorithm] <= p_value_ranges[1]].shape[0] / num_records
                percentage = 100 * ratio
                ranges[benchmark].append(percentage)

            mode_df = pd.DataFrame(ranges)
            mode_df['Mean'] = mode_df.mean(numeric_only=True, axis=1)
            mode_df = mode_df.round(2)
            print(mode)
            if report_once:
                print(','.join([str(item) for item in mode_df.columns]))
                report_once = False
            for idx, row in mode_df.iterrows():
                print('& ' + ' & '.join([str(item) for item in row]))
            print()
        logger.info("-------------------------------------------------")

    def plot_medians_difference(self, cfg: dict, bma: BenchmarkAnalysis.__class__,
                                bmap: BenchmarkAnalysisPlots.__class__):

        logger.info("Plotting medians difference %")

        def _run(benchmarks, prefix=None):
            # Start plotting
            bap: BenchmarkAnalysisPlots = bmap(cfg)
            bap.benchmarks = benchmarks
            for _bm in bap.benchmarks:
                for metric in cfg[S_METRICS]:
                    df = self.benchmarks_analysis(_bm, metric, cfg, bma)
                    # bap.intra_node_median_difference(_bm, metric, df, prefix=prefix)
                    # bap.inter_node_median_difference(_bm, metric, df, prefix=prefix)
                    bap.medians_difference(_bm, metric, df, prefix)

        individual_plotting = cfg[S_INDIVIDUAL_PLOTS] if S_INDIVIDUAL_PLOTS in cfg.keys() else True
        if individual_plotting:
            # If individual benchmark plotting use this
            for bm in cfg[S_BENCHMARKS]:
                _run([bm], bm)
        else:
            # Combined plotting use this
            n = len(cfg[S_BENCHMARKS])

            # Plot k benchmarks per figure
            k = 6
            i, j = divmod(n, k)
            for c in range(i):
                _run(benchmarks=cfg[S_BENCHMARKS][c * k:c * k + k], prefix=c + 1)

            if j != 0:
                _run(benchmarks=cfg[S_BENCHMARKS][-j:], prefix=i + 1)

    def main(self, cfg, bma, bmap):
        # Run this for medians plots
        self.plot_medians_difference(cfg, bma, bmap)

        # Run this for statistics
        self.benchmarks_statistical_analysis(cfg, bma)
