import numpy as np
import pandas as pd
from scipy import stats

from pyutils.analysis.variability.statistical_analysis.effect_size import EffectSize, ROBUST_D
from pyutils.analysis.variability.statistical_analysis.strings import KS_PVALUE, KS_STATISTIC
from pyutils.common.strings import S_NODE_ID


def _two_samples_tests(gn, gd, metric_key, key_to_dict):
    """ Method that can be run in parallel on groups of a data frame to calculate
    2-samples tests. """
    results = list()
    nodes = gd[S_NODE_ID].values.data

    # All permutations
    node_pairs = list()
    n = len(nodes)
    for i in range(n - 1):
        for j in range(i + 1, n):
            node_pairs.append([nodes[i], nodes[j]])
    # logger.info(f"-- {len(node_pairs)} node pairs.")

    key = '_'.join([str(i) for i in gn])
    config = key_to_dict(key)

    for node_a_idx, node_b_idx in node_pairs:
        # TODO: we use 0 here assuming we have one array per node. Otherwise, the groupby will violate this.
        node_a_data = gd[gd[S_NODE_ID] == node_a_idx][metric_key].values[0]
        node_b_data = gd[gd[S_NODE_ID] == node_b_idx][metric_key].values[0]
        ks_result = stats.ks_2samp(node_a_data, node_b_data)._asdict()
        robust_d_val = EffectSize.scaled_robust_d(node_a_data, node_b_data)
        summary = {
            # CONFIG_ID: config_id,
            'node_a': node_a_idx, 'node_b': node_b_idx,
            'mean_a': np.mean(node_a_data), 'mean_b': np.mean(node_b_data),
            'std_a': np.std(node_a_data), 'std_b': np.std(node_b_data),
            # 'cohen_d': d_cohen_val,
            ROBUST_D: robust_d_val,
            KS_PVALUE: ks_result['pvalue'], KS_STATISTIC: ks_result['statistic']
            # 'spearman_corr': spearman_corr, 'spearman_pvalue': spearman_pvalue
        }
        results.append({**summary, **config})
    return pd.DataFrame(results)
