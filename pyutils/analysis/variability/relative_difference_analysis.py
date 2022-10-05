import os

import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd, delayed


from pyutils.analysis.variability.parallel_utils import get_benchmark_data_for_nodes, \
    par_calculate_and_set_group_median, \
    mapped_calculate_inter_and_intra_variability, dask_remove_variability_outliers
from pyutils.analysis.variability.statistical_analysis.hypothesis_tests import HypothesisTests
from pyutils.analysis.variability.strings import S_CURRENT_TIMESTAMP, S_MEDIAN_OF_ALL, S_INTRA, S_INTER
from pyutils.analysis.variability.vision_networks.variability import NetworkAnalysisMethods
from pyutils.characterization.networks.utils import is_network
from pyutils.common.config import SCHEDULER, S_DVFS_COLS, N_ROWS_PER_PARTITION
from pyutils.common.data_handlers.data_interface import DataHandler
from pyutils.common.paths import CROSS_NODE_ANALYSIS
from pyutils.common.strings import S_BENCHMARK, S_METRIC, S_LIMIT_NUM_OBSERVATIONS, S_MIN_NUM_OBSERVATIONS, \
    S_KEEP_SUBMISSION_TIMESTAMP, S_NODE_ID, S_SUBMISSION_TIMESTAMP, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ
from pyutils.common.utils import prepare, FileUtils, GlobalLogger
from pyutils.run.analysis.config import S_BENCHMARKS, S_NODES, S_RECALCULATE_MEDIANS, \
    S_RECALCULATE_MEDIANS_DIFF, S_ABS_DIFF

logger = GlobalLogger().get_logger()


class BenchmarkAnalysis:
    """ A class that implements methods for statistical and empirical analysis of the experimental data
    of Rodinia and other benchmarks. The goal is to mainly compare inter and intra-node variability
    magnitudes.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.benchmark = cfg[S_BENCHMARK]
        self.benchmark_output_dir = prepare(CROSS_NODE_ANALYSIS, S_BENCHMARKS, self.benchmark)
        self.medians_file = prepare(self.benchmark_output_dir, cfg[S_METRIC],
                                    f'{self.benchmark}_medians.parquet')
        self.medians_diff_file = prepare(self.benchmark_output_dir, cfg[S_METRIC],
                                         f'{self.benchmark}_medians_diff.parquet')
        self.data_with_outliers_removed = prepare(self.benchmark_output_dir, cfg[S_METRIC],
                                                  f'{self.benchmark}_medians_diff_with_outliers_removed.parquet')
        self.metric = cfg[S_METRIC]

        if is_network(self.benchmark):
            self.bm_methods = NetworkAnalysisMethods
        else:
            self.bm_methods = None
        self.get_bm_data_method = DataHandler.get_benchmark_data
        if cfg[S_LIMIT_NUM_OBSERVATIONS]:
            self.min_num_obs = cfg[S_MIN_NUM_OBSERVATIONS][self.metric]
            logger.info(f"Limiting number of observations to {self.min_num_obs}.")
        else:
            self.min_num_obs = None
            logger.info("Not limiting number of observations.")

        self.keep_submission_timestamp = S_KEEP_SUBMISSION_TIMESTAMP in self.cfg.keys()
        logger.info(f"Submission timestamp-based analysis: {self.keep_submission_timestamp}")

    def statistical_analysis(self, overwrite_data=False):
        """ Calculate the k-AD and 2-samples hypothesis tests results for the given metric.
        @returns: None, the output is the two CSV files containing the calculated stats.
        """
        # Prepare output directories and files
        metric = self.cfg[S_METRIC]
        metric_key = DataHandler.get_metric_key(self.metric)
        metric_output_dir = prepare(self.benchmark_output_dir, metric)
        k_ad_test_output_file = prepare(metric_output_dir, 'k_samples_ad.parquet')
        two_samples_tests_output_file = prepare(metric_output_dir, 'two_samples_ks.parquet')
        corrected_two_samples_tests_output_file = prepare(metric_output_dir,
                                                          'corrected_two_samples_ks.parquet')

        def execute_hypothesis_tests(_data):
            if os.path.exists(k_ad_test_output_file) and not overwrite_data:
                k_ad_df = FileUtils.deserialize(k_ad_test_output_file)
            else:
                # K-ad tests
                # TODO: better handling of rodinia keys because we run them with different kernel parameters
                # so, the freqs and benchmark name are not enough as identifiers.

                k_ad_df = HypothesisTests.k_ad_on_df(_data, metric_key, self.bm_methods.create_dict_from_key)
                k_ad_df[S_BENCHMARK] = self.benchmark
                k_ad_df[S_METRIC] = metric
                FileUtils.serialize(k_ad_df, k_ad_test_output_file)

            if os.path.exists(two_samples_tests_output_file) and not overwrite_data:
                two_samples_ks_df = FileUtils.deserialize(two_samples_tests_output_file)
            else:
                # Two samples tests
                two_samples_ks_df = HypothesisTests.two_samples_tests_on_df(df, metric_key,
                                                                            self.bm_methods.create_dict_from_key)
                FileUtils.serialize(two_samples_ks_df, two_samples_tests_output_file)

            if os.path.exists(corrected_two_samples_tests_output_file) and not overwrite_data:
                corrected_2ks_df = FileUtils.deserialize(corrected_two_samples_tests_output_file)
            else:
                corrected_2ks_df = HypothesisTests.fdr_correction(two_samples_ks_df)
                corrected_2ks_df[S_BENCHMARK] = self.benchmark
                corrected_2ks_df[S_METRIC] = metric
                FileUtils.serialize(corrected_2ks_df, corrected_two_samples_tests_output_file)
            return k_ad_df, corrected_2ks_df

        df = self.calculate_local_median()
        if type(df) is not pd.DataFrame:
            df: pd.DataFrame = df.compute(scheduler=SCHEDULER)

        return execute_hypothesis_tests(df)

    def calculate_local_median(self) -> pd.DataFrame:
        """ Calculates the median of observations per node on a per-configuration basis. The resultant DF
        is used for
        intra and inter-node variability analysis. Usually is followed by @calculate_global_median.
        @returns: df, a dataframe object contains the benchmark parameters, and DVFS configs along with the
        median per node for the specified metric.
        """
        if os.path.exists(self.medians_file) and not self.cfg[S_RECALCULATE_MEDIANS]:
            logger.info(f"Loading cached medians file: {self.medians_file}")
            return dd.read_parquet(self.medians_file, split_row_groups=N_ROWS_PER_PARTITION)

        logger.info(f"Calculating local medians...")

        metric_key = DataHandler.get_metric_key(self.metric)

        logger.info("-- Parse benchmarks data.")
        d_cfg = delayed(self.cfg)

        # Parse data for all timestamps
        res = dask.compute(get_benchmark_data_for_nodes(self.cfg[S_NODES], d_cfg, self.get_bm_data_method,
                                                        metric_key, self.min_num_obs), scheduler=SCHEDULER)
        df = dd.concat(res[0])
        if type(df) is not pd.DataFrame:
            df = df.compute(scheduler=SCHEDULER)
        df = df.sort_values(by=[S_BENCHMARK, S_NODE_ID] + S_DVFS_COLS)
        df = df.reset_index(drop=True)
        return df

    def calculate_global_median(self, df):
        """ Calculates the median of medians (Global). The resultant DF is used for intra and inter-node
        variability
        analysis.
        @returns: df, a dataframe object contains the benchmark parameters, and DVFS configs along with the
        median per node and median of nodes values for the specified metric.
        """
        logger.info(f"Calculating global medians...")
        if self.keep_submission_timestamp:
            # Virtually consider each timestamp as a separate benchmark, so we select only one timestamp to
            # process at
            # a time. The driver class will set this current timestamp in a loop.
            df = df[df[S_SUBMISSION_TIMESTAMP] == self.cfg[S_CURRENT_TIMESTAMP]]

        # Group by all keys except node id to create a median
        group_by_cols = [S_BENCHMARK]

        logger.info("-- Calculate and set global medians")
        results = [delayed(par_calculate_and_set_group_median)(group_df, S_MEDIAN_OF_ALL) for _, group_df in
                   df.groupby(group_by_cols)]
        t = dask.compute(results, scheduler=SCHEDULER)
        df = dd.concat(t[0])

        logger.info(f"-- Done. Data shape is: {df.shape}. Saving to: {self.medians_file}")
        FileUtils.serialize(df, self.medians_file)
        return dd.read_parquet(self.medians_file, split_row_groups=N_ROWS_PER_PARTITION)

    def calculate_all_medians(self):
        df = self.calculate_local_median()
        df = self.calculate_global_median(df)
        return df

    def calculate_medians_difference(self):
        """ Calculates the % difference between observations and medians (i.e., per-node medians,
        and median-of-medians)
        Later, we plot this % to compare intra-node to inter-node variability. """

        if os.path.exists(self.medians_diff_file) and not self.cfg[S_RECALCULATE_MEDIANS_DIFF] and \
                not self.cfg[S_RECALCULATE_MEDIANS]:
            logger.info(f"Loading medians difference % from: {self.medians_diff_file}")
            return dd.read_parquet(self.medians_diff_file, split_row_groups=N_ROWS_PER_PARTITION)

        if not os.path.exists(self.medians_file) or self.cfg[S_RECALCULATE_MEDIANS]:
            df = self.calculate_all_medians()
        else:
            logger.info(f"Loading medians file at {self.medians_file}")
            df: dd.DataFrame = dd.read_parquet(self.medians_file, split_row_groups=N_ROWS_PER_PARTITION)

        logger.info("Calculating medians difference %.")
        metric_key = DataHandler.get_metric_key(self.cfg[S_METRIC])
        df = df.map_partitions(mapped_calculate_inter_and_intra_variability, metric_key, self.cfg[S_ABS_DIFF],
                               meta={S_CPU_FREQ: np.uint8, S_GPU_FREQ: np.uint8, S_MEMORY_FREQ: np.uint8,
                                     S_NODE_ID: np.uint8, S_INTRA: 'float64', S_INTER: 'float64'})
        df: pd.DataFrame = df.compute(scheduler=SCHEDULER)
        if self.keep_submission_timestamp:
            logger.info(f"Selected timestamp {self.cfg[S_CURRENT_TIMESTAMP]} "
                        f"from {self.benchmark} has {df.shape} data shape.")
            df[S_BENCHMARK] = f'{self.benchmark}_{self.cfg[S_CURRENT_TIMESTAMP]}'
        else:
            df[S_BENCHMARK] = self.benchmark

        df = df.sort_values(by=S_DVFS_COLS + [S_BENCHMARK, S_NODE_ID])
        df = df.reset_index(drop=True)

        FileUtils.serialize(df, self.medians_diff_file)
        return dd.read_parquet(self.medians_diff_file, split_row_groups=N_ROWS_PER_PARTITION)

    def remove_outliers(self, df: pd.DataFrame):
        if os.path.exists(self.data_with_outliers_removed) and not self.cfg[S_RECALCULATE_MEDIANS_DIFF] and \
                not self.cfg[S_RECALCULATE_MEDIANS]:
            logger.info(
                f"Loading medians difference % with removed outliers from: {self.data_with_outliers_removed}")
            return FileUtils.deserialize(self.data_with_outliers_removed)

        logger.info("Removing outliers from medians difference %.")
        df = df.map_partitions(dask_remove_variability_outliers,
                               meta={S_CPU_FREQ: np.uint8, S_GPU_FREQ: np.uint8, S_MEMORY_FREQ: np.uint8,
                                     S_NODE_ID: np.uint8, S_INTRA: 'float64', S_INTER: 'float64',
                                     S_BENCHMARK: str})

        df: pd.DataFrame = df.compute(scheduler=SCHEDULER)
        df = df.reset_index(drop=True)
        FileUtils.serialize(df, self.data_with_outliers_removed)
        return df
