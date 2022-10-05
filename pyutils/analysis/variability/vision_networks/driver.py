import sys

import dask

from pyutils.analysis.variability.benchmark_analysis import Driver
from pyutils.analysis.variability.relative_difference_analysis import BenchmarkAnalysis
from pyutils.analysis.variability.relative_difference_plotting import BenchmarkAnalysisPlots
from pyutils.common.config import MEMORY_GB_PER_WORKER, \
    USE_PROCESSES, N_WORKERS, THREADS_PER_WORKER
from pyutils.common.utils import FileUtils, GlobalLogger
from pyutils.run.analysis.analysis_jobs_config_creator import encapsulate_calls, \
    create_network_variability_jb_cfg


logger = GlobalLogger().get_logger()

if __name__ == '__main__':
    from dask.distributed import Client, LocalCluster

    # os.environ["MALLOC_TRIM_THRESHOLD_"] = '0'

    logger.info(f"Dask server: {N_WORKERS} workers - {THREADS_PER_WORKER} threads and "
                f"{MEMORY_GB_PER_WORKER} GiB memory limit per worker.")
    if MEMORY_GB_PER_WORKER == 0:
        logger.warning(f"Launching with {MEMORY_GB_PER_WORKER} memory. Minimum required is 1 ")
        sys.exit(0)

    cluster = LocalCluster(n_workers=N_WORKERS, processes=USE_PROCESSES,
                           threads_per_worker=THREADS_PER_WORKER, protocol="tcp", ip="192.168.0.20",
                           memory_limit=f'{MEMORY_GB_PER_WORKER}GiB')

    if len(sys.argv) != 2:
        logger.warning("Specify the configuration file first. Creating one for you.")
        config_file_path = encapsulate_calls(create_network_variability_jb_cfg)
    else:
        config_file_path = sys.argv[1]

    logger.info(f"Loading configuration file at: {config_file_path}")
    jb_cfg = FileUtils.deserialize(config_file_path)

    with Client(cluster) as client:
        # Run the main script using the supplied configuration file.
        d = Driver()
        d.main(jb_cfg, BenchmarkAnalysis, BenchmarkAnalysisPlots)

    logger.info("Closing DASK cluster.")
