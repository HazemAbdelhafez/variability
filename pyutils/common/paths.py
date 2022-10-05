from pyutils.common.config import PROJECT_NAME
from pyutils.common.methods import join as jp
from pyutils.common.strings import S_RUNTIME
from pyutils.hosts.common import HOSTNAME, PLATFORM_ARCH


def get_project_dir():
    if PLATFORM_ARCH == 'x86_64':
        if HOSTNAME.__contains__('node260') or HOSTNAME.__contains__('node240') or \
                HOSTNAME.__contains__('node290'):
            project_dir = f"/local/hazem/projects/{PROJECT_NAME}"
        elif HOSTNAME.__contains__('node270'):
            project_dir = f"/nvme0/hazem/{PROJECT_NAME}"
        else:
            project_dir = f"/home/ubuntu/projects/{PROJECT_NAME}"
    else:
        if HOSTNAME == 'node-15':  # node-15
            project_dir = f"/ssd/projects/{PROJECT_NAME}"
        elif HOSTNAME == 'dev':
            project_dir = f"/root/{PROJECT_NAME}"
        else:  # the rest of the nodes
            project_dir = f"/home/ubuntu/projects/{PROJECT_NAME}"
    return project_dir


PROJECT_DIR = get_project_dir()

MODELING_JOB_STATUS_DIR = "/home/ubuntu/nodes-status"
CHARACTERIZATION_JOB_STATUS_DIR = "/home/ubuntu/nodes-characterization-status"
VISION_NETWORKS_JOB_STATUS = "/home/ubuntu/nodes-status/vision-networks"

DATA_DIR = jp(PROJECT_DIR, "data")
RESOURCES_DATA_DIR = jp(DATA_DIR, "resources")
DVFS_CONFIGS_DIR = jp(RESOURCES_DATA_DIR, "dvfs-configs")

KERNELS_RESOURCES_DATA_DIR = jp(RESOURCES_DATA_DIR, "kernels")
VISION_NETWORKS_PROFILING_OUTPUT_DIR = jp(DATA_DIR, "vision-models-profiles")
VISION_KERNELS_PROFILING_OUTPUT_DIR = jp(VISION_NETWORKS_PROFILING_OUTPUT_DIR, 'kernels-profiles')
VISION_KERNELS_RUNTIME_PROFILING_OUTPUT_DIR = jp(VISION_KERNELS_PROFILING_OUTPUT_DIR, 'runtime')
VISION_KERNELS_POWER_PROFILING_OUTPUT_DIR = jp(VISION_KERNELS_PROFILING_OUTPUT_DIR, 'power')
TMP_DIR = jp(DATA_DIR, "temp")
SCRIPTED_MODULES_DIR = jp(DATA_DIR, "scripted-modules", "final")
CONV2D_PROFILING_DIR = jp(DATA_DIR, "conv2d-profiling")
GENERATED_MANUAL_ANALYSIS = jp(DATA_DIR, "manual-analysis")
SAVED_MODELS_DIR = jp(DATA_DIR, "saved-models")
ONE_VS_MANY_MODELS_DIR = jp(DATA_DIR, SAVED_MODELS_DIR, 'one_vs_many')
MODELS_PERFORMANCE_DIR = jp(DATA_DIR, 'models-performance')
FIGURES_DIR = jp(DATA_DIR, 'figures')
PREDICTION_FIGURES_DIR = jp(FIGURES_DIR, 'prediction')
MODELS_PERFORMANCE_FIGURES_DIR = jp(FIGURES_DIR, 'models-performance')
WATSSUP_PATH = jp(f'{DATA_DIR}', 'bin', 'wattsup')
CHARACTERIZATION_DATA_DIR = jp(DATA_DIR, "characterization")
MODELING_DATA_DIR = jp(DATA_DIR, "modeling-data")
NVPROF_OUTPUT_PARENT_DIR = jp(CONV2D_PROFILING_DIR, "nvprof-output")
NVPROF_ANALYSIS_PARENT_OUTPUT_DIR = jp(CONV2D_PROFILING_DIR, "nvprof-analysis")
NVPROF_SUMMARY_PARENT_OUTPUT_DIR = jp(CONV2D_PROFILING_DIR, "nvprof-summary")
NVPROF_OUTPUT_DIR = NVPROF_OUTPUT_PARENT_DIR
NVPROF_ANALYSIS_OUTPUT_DIR = NVPROF_ANALYSIS_PARENT_OUTPUT_DIR
NVPROF_SUMMARY_OUTPUT_DIR = jp(NVPROF_SUMMARY_PARENT_OUTPUT_DIR, HOSTNAME)
NETWORKS_CHARACTERIZATION_DIR = jp(DATA_DIR, 'networks-characterization')
JOBS_CFG_DIR = jp(DATA_DIR, 'jobs')
PREDICTION_DIR = jp(DATA_DIR, 'prediction')
NETWORKS_PREDICTION_DIR = jp(PREDICTION_DIR, 'networks')
RUNTIME_PREDICTION_DIR = jp(PREDICTION_DIR, S_RUNTIME)
STACK_TRACE_REPLAY_DIR = jp(DATA_DIR, "stack-traces-replays")
LOGGING_DIR = jp(DATA_DIR, "logs")
CACHE_DIR = jp(DATA_DIR, "cache")

CROSS_NODE_ANALYSIS = jp(DATA_DIR, 'cross-node-analysis')
CROSS_NODE_ANALYSIS_STATISTICS = jp(CROSS_NODE_ANALYSIS, 'statistics')

CROSS_NODE_ANALYSIS_TIMING_STATISTICS = jp(CROSS_NODE_ANALYSIS_STATISTICS, 'timing')
CROSS_NODE_ANALYSIS_POWER_STATISTICS = jp(CROSS_NODE_ANALYSIS_STATISTICS, 'power')
