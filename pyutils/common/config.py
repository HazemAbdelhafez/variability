import datetime
import functools
import os

from pyutils.common.strings import S_RUNTIME_MS, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, S_AVG_PWR_W, \
    S_MEASURED_ENERGY_J, S_CALCULATED_ENERGY_J
from pyutils.hosts.common import PLATFORM_ARCH

PROJECT_NAME = "variability"


SPLIT_SIZE = 100
DVFS_SPACE_SIZE = 3654
CONV2D_SPACE_SIZE = 7055
GENERIC_SPACE_SIZE = 10000
MAX_NUM_RECORDS = 1000
IGNORED_TIMING_RECORDS = 10

FRONT_IGNORED_POWER_RECORDS = 20
TAIL_IGNORED_POWER_RECORDS = 5
DEFAULT_PWR_MAX_ITERATIONS = 500
TMP_PWR_ITERATIONS = 50

# Experiments' parameters
WARM_UP_ITERATIONS = 5
TIMING_ITERATIONS = 50
UPPER_LATENCY_THRESHOLD = 1000  # Time in ms

ORIGINAL_LABELS = [S_RUNTIME_MS, S_AVG_PWR_W, S_CALCULATED_ENERGY_J, S_MEASURED_ENERGY_J]

ALL_LABELS = ORIGINAL_LABELS

# Set global timestamp
DAY_TIMESTAMP = str(datetime.datetime.now().strftime("%Y%m%d"))
HOUR_TIMESTAMP = str(datetime.datetime.now().strftime("%Y%m%d%H"))
MINUTE_TIMESTAMP = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))

N_CPU_CORES = os.cpu_count() - 4

if PLATFORM_ARCH == 'x86_64':
    # PSUTIL is not installed on the jetson boards
    import psutil

    MEMORY_LIMIT = int(psutil.virtual_memory().available * 0.8)
else:
    MEMORY_LIMIT = 1 * 1024 * 1024 * 1024

MEMORY_LIMIT_GB = int(MEMORY_LIMIT / (1024 * 1024 * 1024))


def factors(_n):
    seq = ([i, _n // i] for i in range(1, int(pow(_n, 0.5) + 1)) if _n % i == 0)
    return set(functools.reduce(list.__add__, seq))


MEMORY_GB_PER_WORKER = 8
N_WORKERS = max(1, min(int(N_CPU_CORES / 20), int(MEMORY_LIMIT / MEMORY_GB_PER_WORKER)))
THREADS_PER_WORKER = 20
N_ROWS_PER_PARTITION = 30
N_PARTITIONS = (3 * THREADS_PER_WORKER * N_WORKERS)

# USE_PROCESSES = True if 20 > N_JOBS > 10 else False
USE_PROCESSES = False

if USE_PROCESSES:
    SCHEDULER = 'dask.distributed'
else:
    SCHEDULER = 'dask.distributed'


class Timers:
    class CUEvents:
        per_run = 'cuda_event_per_run'
        per_run_with_preload = 'per_run_with_preload'
        sync_once = 'cuda_sync_once'
        nvprof = 'nvprof'
        block_based = "block_based"
        cuda_graphs = "cuda_graphs"
        overall = "overall"


class PowerMeasurementMethods:
    sync_once = "sync_once"
    block_based = "block_based"


THIRTEEN_NODES = ['node-%02d' % i for i in range(1, 14)]
ALL_NODES = THIRTEEN_NODES + ['node-14']
MIN_CPU_FREQ = 422400

S_DVFS_COLS = [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ]

# Precision of measurements configs
RME_THRESHOLD = 0.5
CONFIDENCE_LVL = 99
DUMMY_RESTARTS = 5
MIN_NUM_OF_VALID_RESTARTS = 50

# Experiments timing methods
ALL_CONTROLS = 'all_controls'
TIMING_MEASUREMENT_METHOD_1 = '1'
POWER_MEASUREMENT_METHOD_1 = '1'
POWER_MEASUREMENT_METHOD_2 = '2'

JIT_FUSER = "fuser0"  # if this is not set to zero, you must use the two values below in the
# context managers.
JIT_FUSE_ON_CPU = True
JIT_FUSE_ON_GPU = True
