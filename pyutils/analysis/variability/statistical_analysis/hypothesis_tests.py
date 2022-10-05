import multiprocessing

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, anderson
from scipy.stats.morestats import ShapiroResult, AndersonResult
from statsmodels.stats.multitest import multipletests

from pyutils.analysis.variability.statistical_analysis.effect_size import EffectSize, ROBUST_D
from pyutils.analysis.variability.statistical_analysis.strings import KS_PVALUE, \
    BONFERRONI_CORRECTED_KS_PVALUE, BH_CORRECTED_KS_PVALUE, KS_STATISTIC, AD_k_STATISTIC, AD_k_PVALUE, \
    CONFIG_ID
from pyutils.analysis.variability.statistical_analysis.utils import _two_samples_tests
from pyutils.common.config import S_DVFS_COLS
from pyutils.common.strings import S_BENCHMARK
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()


class HypothesisTests:
    @staticmethod
    def k_ad_on_df(df: pd.DataFrame, column, key_to_dict) -> pd.DataFrame:
        result = list()
        for gn, gd in df.groupby(by=S_DVFS_COLS + [S_BENCHMARK]):
            ks_result = stats.anderson_ksamp(gd[column])._asdict()
            key = '_'.join([str(i) for i in gn])
            tmp = {**key_to_dict(key),
                   **{
                       AD_k_STATISTIC: ks_result['statistic'],
                       AD_k_PVALUE: ks_result['significance_level'],
                       'critical_value_at_1%': ks_result['critical_values'][-1],
                       'critical_value_at_5%': ks_result['critical_values'][-2],
                       "num_valid_nodes_arrays": len(gd[column])}}
            result.append(tmp)
        return pd.DataFrame(result)

    @staticmethod
    def k_ad_on_dict(data, key_to_dict):
        configs = list(data.keys())
        all_records = list()
        for i in range(len(configs)):
            config_key = configs[i]

            # For each config we have number of samples k == number of nodes
            k_samples = list()
            for sample in data[config_key]:
                k_samples.append(sample)

            ks_result = stats.anderson_ksamp(k_samples)._asdict()
            all_records.append({**key_to_dict(config_key),
                                **{AD_k_STATISTIC: ks_result['statistic'],
                                   AD_k_PVALUE: ks_result['significance_level'],
                                   'critical_value_at_1%': ks_result['critical_values'][-1],
                                   'critical_value_at_5%': ks_result['critical_values'][-2]
                                   }})

            # if i == 0:
            #     logger.info(f"K-AD critical values for {len(k_samples)} is: {ks_result['critical_values']}")

        df = pd.DataFrame(all_records)
        return df

    @staticmethod
    def two_samples_tests_on_dict(data, key_to_dict) -> pd.DataFrame:
        configs = list(data.keys())
        n = len(configs)
        num_nodes = len(list(data.values())[0])
        for i in data.values():
            if len(i) != num_nodes:
                raise Exception(f"Un matched number of nodes in combined data set: {len(i)} != {num_nodes}")

        # All permutations
        node_pairs = list()
        for a in range(num_nodes - 1):
            for b in range(a + 1, num_nodes):
                node_pairs.append([a, b])
        logger.info(f"Running 2-Samples tests: ")
        logger.info(f"-- {len(node_pairs)} node pairs.")
        logger.info(f"-- {int(n)} configs.")

        all_records = list()
        counter = 0
        for node_a_idx, node_b_idx in node_pairs:
            config_id = 0
            for config_key in configs:
                node_a_data = data[config_key][node_a_idx]
                node_b_data = data[config_key][node_b_idx]
                config = key_to_dict(config_key)

                # Stats
                ks_result = stats.ks_2samp(node_a_data, node_b_data)._asdict()

                # d_cohen_val = EffectSize.cohen_d(node_a_data, node_b_data)
                # spearman_corr, spearman_pvalue = spearmanr(node_a_data, node_b_data)
                robust_d_val = EffectSize.scaled_robust_d(node_a_data, node_b_data)

                summary = {
                    CONFIG_ID: config_id,
                    'node_a': node_a_idx, 'node_b': node_b_idx,
                    'mean_a': np.mean(node_a_data), 'mean_b': np.mean(node_b_data),
                    'std_a': np.std(node_a_data), 'std_b': np.std(node_b_data),
                    # 'cohen_d': d_cohen_val,
                    ROBUST_D: robust_d_val,
                    KS_PVALUE: ks_result['pvalue'], KS_STATISTIC: ks_result['statistic']
                    # 'spearman_corr': spearman_corr, 'spearman_pvalue': spearman_pvalue
                }
                all_records.append({**summary, **config})

                config_id += 1

            # Progress counter
            if counter % 10 == 0:
                logger.info(f"At pair: {counter}")

            counter += 1
        df = pd.DataFrame(all_records)
        return df

    @staticmethod
    def two_samples_tests_on_df(df: pd.DataFrame, metric_key: str, key_to_dict) -> pd.DataFrame:

        logger.info(f"Running 2-Samples tests on dataframe of size: {df.shape}")

        # TODO: update this parameter
        pool = multiprocessing.Pool(processes=24)

        results = list()
        for _gn, _gd in df.groupby(by=S_DVFS_COLS + [S_BENCHMARK]):
            res = pool.apply_async(_two_samples_tests, (_gn, _gd, metric_key, key_to_dict))
            results.append(res.get())
        return pd.concat(results)

    @staticmethod
    def fdr_correction(ks_test_results: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the p-values and correct for the many tests. There are two choices: 1) correct on a per
        config basis,
        or 2) correct on all configs at once on a per benchmark basis.
        The second is more conservative because the corrected p-value will depend on the number of tests (
        the larger,
        the smaller the threshold). So we pick it (this is inline with the statistical consultation from
        UBC stats dep.)
        """

        # It doesn't matter the alpha value here, we are only interested in the corrected pvalues:
        # Quote from docs ->
        #     Except for 'fdr_twostage', the p-value correction is independent of the
        #     alpha specified as argument. In these cases the corrected p-values
        #     can also be compared with a different alpha. In the case of 'fdr_twostage',
        #     the corrected p-values are specific to the given alpha, see
        #     ``fdrcorrection_twostage``.
        alpha_does_not_matter = 0.05  # Read comment above from library documentation.

        ks_test_results[BH_CORRECTED_KS_PVALUE] = multipletests(ks_test_results[KS_PVALUE],
                                                                alpha=alpha_does_not_matter,
                                                                method='fdr_bh')[1]
        ks_test_results[BONFERRONI_CORRECTED_KS_PVALUE] = multipletests(ks_test_results[KS_PVALUE],
                                                                        alpha=alpha_does_not_matter,
                                                                        method='bonferroni')[1]
        return ks_test_results

    @staticmethod
    def normality_test_on_df(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
        """ Applies test of normality on the metric data array directly. """

        def is_norm_with_shapiro(x: pd.Series, alpha):
            """ If normal, return true, else false"""
            tmp: ShapiroResult = shapiro(x)
            return tmp.pvalue > alpha

        def is_norm_with_anderson(x: pd.Series, alpha):
            """ If normal, return true, else false"""
            tmp: AndersonResult = anderson(x, dist="norm")
            # Get index of critical values corresponding to the given alpha level
            idx = list(tmp.significance_level).index(alpha * 100)
            critical_value = tmp.critical_values.item(idx)
            return tmp.statistic < critical_value

        for i in [0.01, 0.05]:
            df[f"Shapiro-Wilk-{int(i * 100)}%"] = df[metric_key].apply(is_norm_with_shapiro, args=(i,))
            df[f"Anderson-Darling-{int(i * 100)}%"] = df[metric_key].apply(is_norm_with_anderson, args=(i,))
        return df


