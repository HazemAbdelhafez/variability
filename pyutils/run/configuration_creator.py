import sys
from os.path import join as join_path

from pyutils.characterization.networks.factory import NetworksLoaders
from pyutils.characterization.networks.utils import get_unified_benchmark_name
from pyutils.common.config import Timers
from pyutils.common.paths import JOBS_CFG_DIR
from pyutils.common.strings import S_METRIC, S_RUNTIME, S_BENCHMARK, \
    S_EXP_NOTES, S_WARMUP_ITERATIONS, S_PWR_SMPLG_RATE, \
    S_INTERNAL_PWR_METER, S_TIMING_METHOD, S_CORE_AFFINITY, \
    S_NUM_ISOLATED_CORES, S_CONTROL_SCENARIO, S_NUM_OBSERVATIONS, S_BLOCK_RUNTIME_MS, S_BLOCK_SIZE, S_RME, \
    S_CONFIDENCE_LVL, S_RECORD_DATA, S_DVFS_CONFIG_FILE_PATH, S_PWR_METER, S_SUBMISSION_TIMESTAMP, \
    S_SKIP_SEEN_BEFORE, \
    S_NUM_RESTARTS, S_DISCARDED_RESTARTS, \
    S_EXP_TIMEOUT_DURATION, S_MAX_NUM_RETRIES, S_GRAPH_SIZE, S_CHECK_RME, S_CHARACTERIZE_CHILDREN_KERNELS, \
    S_POWER, \
    S_PARAMETERS, S_DISABLE_ASLR
from pyutils.common.utils import FileUtils, TimeStamp, MHDMY_DATE_FORMAT, READABLE_MHDMY_DATE_FORMAT
from pyutils.run.config import S_DVFS_CONFIGS
from pyutils.run.config import S_THREADS

HOUR = 60 * 60
DAY = 24 * HOUR
MINUTE = 60


def create_vision_networks_job_config(bm='mobilenetv2'):
    job_config = dict()

    job_config[S_BENCHMARK] = get_unified_benchmark_name(bm)
    job_config[S_PARAMETERS] = NetworksLoaders.load_parameters(job_config.get(S_BENCHMARK)).to_dict()
    job_config[S_EXP_NOTES] = "runtime_v1"

    job_config[S_METRIC] = S_RUNTIME
    job_config[S_SKIP_SEEN_BEFORE] = False
    job_config[S_CHARACTERIZE_CHILDREN_KERNELS] = False

    job_config[S_DISABLE_ASLR] = True

    job_config = add_runtime_config(job_config)
    job_config = add_power_config(job_config)
    job_config = add_common_experiment_settings(job_config)
    job_config = add_core_pinning_config(job_config)

    job_config = add_dvfs_config(job_config, num_configs=455)
    save(job_config, 'characterization_job_cfg.json')


def create_network_kernels_job_config(bm='mobilenetv2'):
    job_config = dict()

    job_config[S_BENCHMARK] = get_unified_benchmark_name(bm)
    job_config[S_EXP_NOTES] = "kernels_v1"

    job_config[S_METRIC] = S_RUNTIME
    job_config[S_SKIP_SEEN_BEFORE] = True
    job_config[S_CHARACTERIZE_CHILDREN_KERNELS] = True

    job_config = add_runtime_config(job_config)
    job_config = add_power_config(job_config)
    job_config = add_common_experiment_settings(job_config)
    job_config = add_core_pinning_config(job_config)

    job_config = add_dvfs_config(job_config, num_configs=3)

    save(job_config, 'characterization_job_cfg.json')


def add_runtime_config(job_config: dict):
    if job_config[S_METRIC] == S_RUNTIME:
        job_config[S_TIMING_METHOD] = Timers.CUEvents.cuda_graphs
    return job_config


def add_power_config(job_config: dict):
    if job_config[S_METRIC] == S_POWER:
        job_config[S_PWR_METER] = S_INTERNAL_PWR_METER
        job_config[S_PWR_SMPLG_RATE] = 1
    return job_config


def add_common_experiment_settings(job_config: dict):
    job_config[S_EXP_TIMEOUT_DURATION] = 1800  # Seconds
    job_config[S_MAX_NUM_RETRIES] = 3
    job_config[S_NUM_RESTARTS] = 1

    job_config[S_WARMUP_ITERATIONS] = 5
    job_config[S_NUM_OBSERVATIONS] = 35

    job_config[S_DISCARDED_RESTARTS] = 5
    job_config[S_RECORD_DATA] = True

    job_config[S_BLOCK_SIZE] = 100
    job_config[S_GRAPH_SIZE] = job_config.get(S_BLOCK_SIZE)
    job_config[S_BLOCK_RUNTIME_MS] = 1000

    job_config[S_RME] = 0.5
    job_config[S_CHECK_RME] = True
    job_config[S_CONFIDENCE_LVL] = 99

    job_config[S_SUBMISSION_TIMESTAMP] = TimeStamp.get_timestamp(date_format=READABLE_MHDMY_DATE_FORMAT)
    return job_config


def add_dvfs_config(job_config: dict, num_configs: int):
    job_config[S_DVFS_CONFIG_FILE_PATH] = f'dvfs_{num_configs}.json'
    job_config[S_DVFS_CONFIGS] = list(range(num_configs))

    return job_config


def add_core_pinning_config(job_config: dict, control_scenario=6):
    job_config[S_CONTROL_SCENARIO] = control_scenario
    if job_config[S_CONTROL_SCENARIO] >= 3:
        job_config[S_CORE_AFFINITY] = True
        job_config[S_NUM_ISOLATED_CORES] = 6
        job_config[S_THREADS] = job_config[S_NUM_ISOLATED_CORES]
    else:
        job_config[S_CORE_AFFINITY] = False
        job_config[S_NUM_ISOLATED_CORES] = 0
        job_config[S_THREADS] = 8
    return job_config


def save(job_config: dict, file_name: str):
    FileUtils.serialize(file_path=join_path(JOBS_CFG_DIR, file_name), user_data=job_config)


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        _bm = str(sys.argv[1])
    else:
        _bm = "mobilenetv2"

    if len(sys.argv) >= 3:
        _target = str(sys.argv[2])
    else:
        _target = 'networks'

    if _target == "networks":
        create_vision_networks_job_config(_bm)
    elif _target == "kernels":
        create_network_kernels_job_config(_bm)
    else:
        raise Exception("Specify networks of kernels")
