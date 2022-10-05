from os.path import join as jp

from pyutils.characterization.networks.utils import VISION_NETWORKS
from pyutils.common.config import ALL_NODES
from pyutils.common.paths import DATA_DIR, FIGURES_DIR
from pyutils.common.strings import S_RUNTIME, S_POWER, S_KEEP_OBSERVATIONS, S_MIN_NUM_OBSERVATIONS, \
    S_LIMIT_NUM_OBSERVATIONS, \
    S_KEEP_SUBMISSION_TIMESTAMP
from pyutils.common.utils import TimeStamp, READABLE_MHDMY_DATE_FORMAT, FileUtils, prepare
from pyutils.run.analysis.config import S_RUNTIME_NETWORKS, S_PWR_NETWORKS, S_NODES, S_FIGURE_NAME, \
    S_OVERWRITE_COMBINED_DATA, S_OVERWRITE_PREPROCESSED_DATA, S_OVERWRITE_FIGURES, S_METRICS, S_PLOTTING, \
    S_FONT, \
    S_SMALL, S_MEDIUM, S_BIG, S_SUB_PLTS, S_FIG_SIZE, S_SHOWFLIERS, S_WHIS, S_EXCLUDE_LOWEST_CPU_FREQS, \
    S_FIGURES_DIR, S_TIMESTAMP, S_BENCHMARKS, S_OVERWRITE_SUMMARY, S_RECALCULATE_MEDIANS, \
    S_RECALCULATE_MEDIANS_DIFF, S_ABS_DIFF, S_FILTER_BY_TS, S_SELECTED_TS, S_SELECT_DVFS_CONFIG, \
    S_SELECTED_DVFS_CONFIGS, S_OVERWRITE_AGGREGATE, S_OVERWRITE_STATS, S_K_AD_FIGURE_NAME, \
    S_ROBUST_D_FIGURE_NAME, S_INDIVIDUAL_PLOTS

ANALYSIS_JOBS_CONFIGS_DIR = jp(DATA_DIR, 'analysis-jobs')
INTRA_NODE_CONFIGS_DIR = jp(ANALYSIS_JOBS_CONFIGS_DIR, 'intra-node-variability')
INTER_NODE_CONFIGS_DIR = jp(ANALYSIS_JOBS_CONFIGS_DIR, 'inter-node-variability')
BENCHMARKS_VARIABILITY_CONFIGS_DIR = jp(ANALYSIS_JOBS_CONFIGS_DIR, 'benchmarks-variability')


def readable_ts():
    return TimeStamp.get_timestamp(READABLE_MHDMY_DATE_FORMAT)


def encapsulate_calls(call):
    ts = readable_ts()
    jb_cfg, p = call(ts)
    FileUtils.serialize(jb_cfg, file_path=p, indent=2, append=False)
    return p


def update_post_job(p, new_val):
    cfg = FileUtils.deserialize(p)
    if 'post_job' in cfg.keys():
        cfg['post_job'].update(new_val)
    else:
        cfg['post_job'] = new_val

    FileUtils.serialize(cfg, file_path=p, indent=2)
    return p


def create_inter_node_variability_jb_cfg(ts):
    p = jp(INTER_NODE_CONFIGS_DIR, f"inter_node_variability_{ts}.json")
    jb_cfg = dict()
    jb_cfg[S_RUNTIME_NETWORKS] = ['mobilenetv2']
    jb_cfg[S_PWR_NETWORKS] = ["mobilenetv2"]

    jb_cfg[S_NODES] = ALL_NODES
    jb_cfg[S_TIMESTAMP] = ts
    jb_cfg[S_FIGURES_DIR] = jp(FIGURES_DIR, 'inter-node-variability')
    jb_cfg[S_OVERWRITE_COMBINED_DATA] = False
    jb_cfg[S_OVERWRITE_PREPROCESSED_DATA] = True
    jb_cfg[S_OVERWRITE_FIGURES] = False

    jb_cfg[S_SELECT_DVFS_CONFIG] = False
    jb_cfg[S_SELECTED_DVFS_CONFIGS] = []

    jb_cfg[S_METRICS] = [S_RUNTIME, S_POWER]
    jb_cfg[S_EXCLUDE_LOWEST_CPU_FREQS] = True
    jb_cfg[f'{S_RUNTIME}_{S_MIN_NUM_OBSERVATIONS}'] = 90
    jb_cfg[f'{S_POWER}_{S_MIN_NUM_OBSERVATIONS}'] = 50

    # Plotting
    jb_cfg[S_PLOTTING] = dict()
    jb_cfg[S_PLOTTING][S_SHOWFLIERS] = False
    jb_cfg[S_PLOTTING][S_WHIS] = [10, 90]

    return jb_cfg, p


def create_intra_node_variability_jb_cfg(ts):
    p = jp(INTRA_NODE_CONFIGS_DIR, f"intra_node_variability_{ts}.json")
    jb_cfg = dict()
    jb_cfg[S_RUNTIME_NETWORKS] = ['mobilenetv2']
    jb_cfg[S_PWR_NETWORKS] = ['mobilenetv2']

    jb_cfg[S_NODES] = ALL_NODES
    jb_cfg[S_FIGURE_NAME] = f'intra_node_variability_{ts}.png'
    jb_cfg[S_OVERWRITE_AGGREGATE] = False
    jb_cfg[S_OVERWRITE_COMBINED_DATA] = False
    jb_cfg[S_OVERWRITE_PREPROCESSED_DATA] = False
    jb_cfg[S_OVERWRITE_FIGURES] = True

    jb_cfg[S_METRICS] = [S_POWER]
    jb_cfg[S_EXCLUDE_LOWEST_CPU_FREQS] = True

    # Plotting
    jb_cfg[S_PLOTTING] = dict()
    jb_cfg[S_PLOTTING][S_FONT] = dict()
    jb_cfg[S_PLOTTING][S_FONT] = {S_SMALL: 16, S_MEDIUM: 18, S_BIG: 20}
    jb_cfg[S_PLOTTING][S_SUB_PLTS] = {S_FIG_SIZE: (7, 5)}
    jb_cfg[S_PLOTTING][S_SHOWFLIERS] = False
    jb_cfg[S_PLOTTING][S_WHIS] = [10, 90]

    return jb_cfg, p


def create_network_variability_jb_cfg(ts):
    p = prepare(BENCHMARKS_VARIABILITY_CONFIGS_DIR, TimeStamp.parse_timestamp(ts), f"variability_{ts}.json")
    jb_cfg = dict()
    jb_cfg[S_TIMESTAMP] = ts
    jb_cfg[S_BENCHMARKS] = ["mobilenetv2", "squeezenet", "resnet", "alexnet", "shufflenet_v2_x1_0"]
    jb_cfg[S_METRICS] = [S_RUNTIME]
    jb_cfg[S_INDIVIDUAL_PLOTS] = False

    jb_cfg[S_NODES] = ALL_NODES
    jb_cfg[S_FIGURE_NAME] = f'variability_{ts}.png'
    jb_cfg[S_K_AD_FIGURE_NAME] = f'k_ad_{ts}.png'
    jb_cfg[S_ROBUST_D_FIGURE_NAME] = f'robust_d_{ts}.png'

    jb_cfg[S_OVERWRITE_AGGREGATE] = False
    jb_cfg[S_OVERWRITE_COMBINED_DATA] = False
    jb_cfg[S_OVERWRITE_SUMMARY] = False
    jb_cfg[S_RECALCULATE_MEDIANS] = False
    jb_cfg[S_RECALCULATE_MEDIANS_DIFF] = False
    jb_cfg[S_OVERWRITE_STATS] = False

    jb_cfg[S_ABS_DIFF] = False
    jb_cfg[S_EXCLUDE_LOWEST_CPU_FREQS] = True
    jb_cfg[S_LIMIT_NUM_OBSERVATIONS] = False
    jb_cfg[S_MIN_NUM_OBSERVATIONS] = {S_RUNTIME: 50, S_POWER: 50}

    # Benchmark specific configs
    tmp = dict()
    for bm in jb_cfg.get(S_BENCHMARKS):
        tmp[bm] = False
    jb_cfg[S_FILTER_BY_TS] = tmp

    tmp = dict()
    for bm in jb_cfg.get(S_BENCHMARKS):
        tmp[bm] = []
    jb_cfg[S_SELECTED_TS] = tmp

    jb_cfg[S_KEEP_OBSERVATIONS] = True

    # Plotting
    jb_cfg[S_PLOTTING] = dict()
    jb_cfg[S_PLOTTING][S_FONT] = {S_SMALL: 16, S_MEDIUM: 18, S_BIG: 20}
    jb_cfg[S_PLOTTING].update({S_FIG_SIZE: (15, 7)})
    jb_cfg[S_PLOTTING].update({S_WHIS: [0, 100]})
    jb_cfg[S_PLOTTING][S_SHOWFLIERS] = False

    return jb_cfg, p


def create_across_reboots_variability_jb_cfg(ts):
    p = prepare(BENCHMARKS_VARIABILITY_CONFIGS_DIR, f"variability_{ts}.json")
    jb_cfg = dict()
    jb_cfg[S_BENCHMARKS] = ['huffman', 'nw']
    jb_cfg[S_BENCHMARKS] = ['huffman']
    jb_cfg[S_METRICS] = [S_RUNTIME]

    jb_cfg[S_NODES] = ALL_NODES
    jb_cfg[S_FIGURE_NAME] = f'variability_{ts}.png'
    jb_cfg[S_OVERWRITE_AGGREGATE] = False
    jb_cfg[S_OVERWRITE_SUMMARY] = False
    jb_cfg[S_RECALCULATE_MEDIANS] = False
    jb_cfg[S_RECALCULATE_MEDIANS_DIFF] = False
    jb_cfg[S_ABS_DIFF] = False

    # Benchmark specific configs
    jb_cfg[S_FILTER_BY_TS] = {'huffman': True, 'nw': True}
    jb_cfg[S_SELECTED_TS] = {'huffman': [], 'nw': []}

    jb_cfg[S_KEEP_OBSERVATIONS] = True
    jb_cfg[S_KEEP_SUBMISSION_TIMESTAMP] = True

    # Plotting
    jb_cfg[S_PLOTTING] = dict()
    jb_cfg[S_PLOTTING][S_FONT] = {S_SMALL: 16, S_MEDIUM: 18, S_BIG: 20}
    jb_cfg[S_PLOTTING].update({S_FIG_SIZE: (13, 5)})
    jb_cfg[S_PLOTTING].update({S_WHIS: [10, 90]})
    jb_cfg[S_PLOTTING][S_SHOWFLIERS] = False

    return jb_cfg, p


def create_variability_rca_jb_cfg(ts):
    """ Create root cause analysis configuration. """
    p = prepare(BENCHMARKS_VARIABILITY_CONFIGS_DIR, f"variability_{ts}.json")
    jb_cfg = dict()
    jb_cfg[S_BENCHMARKS] = VISION_NETWORKS
    jb_cfg[S_METRICS] = [S_RUNTIME]

    jb_cfg[S_NODES] = ALL_NODES
    jb_cfg[S_FIGURE_NAME] = f'variability_{ts}.png'
    jb_cfg[S_OVERWRITE_AGGREGATE] = False
    jb_cfg[S_OVERWRITE_SUMMARY] = True
    jb_cfg[S_RECALCULATE_MEDIANS] = True
    jb_cfg[S_RECALCULATE_MEDIANS_DIFF] = True
    jb_cfg[S_ABS_DIFF] = False
    jb_cfg[S_EXCLUDE_LOWEST_CPU_FREQS] = True
    jb_cfg[S_MIN_NUM_OBSERVATIONS] = {S_RUNTIME: 95, S_POWER: 25}

    # Benchmark specific configs
    tmp = dict()
    for bm in jb_cfg.get(S_BENCHMARKS):
        tmp[bm] = False
    jb_cfg[S_FILTER_BY_TS] = tmp

    tmp = dict()
    for bm in jb_cfg.get(S_BENCHMARKS):
        tmp[bm] = []
    jb_cfg[S_SELECTED_TS] = tmp

    jb_cfg[S_KEEP_OBSERVATIONS] = True

    # Plotting
    jb_cfg[S_PLOTTING] = dict()
    jb_cfg[S_PLOTTING][S_FONT] = {S_SMALL: 16, S_MEDIUM: 18, S_BIG: 20}
    jb_cfg[S_PLOTTING].update({S_FIG_SIZE: (13, 5)})
    jb_cfg[S_PLOTTING].update({S_WHIS: [10, 90]})
    jb_cfg[S_PLOTTING][S_SHOWFLIERS] = False

    return jb_cfg, p


if __name__ == '__main__':
    pass
