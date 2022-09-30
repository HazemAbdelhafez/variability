from cuda_graphs.utils.timers import TimingConfig, collect_time_observations
from pyutils.characterization.networks.analyzers import stack_trace
from pyutils.characterization.networks.analyzers.runtime import RuntimeAnalyzer
from pyutils.common.config import Timers
from pyutils.common.experiments_utils import TorchSettings
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()


def _time_network_with_cuda_graphs(module, input_obj, timing_itrs):
    c = TimingConfig(warmup_itrs=5, graph_size=5, graph_replays=10)
    out_t = None
    results = [0.0 for _ in range(timing_itrs)]
    warmup = True
    for i in range(timing_itrs):
        elapsed_time_ms, out_t = collect_time_observations(c, module, input_obj, warmup)
        if warmup:
            warmup = False
        results[i] = elapsed_time_ms
    return results, out_t


def run(network):
    TorchSettings.set_optimization_config()
    kernels = stack_trace.get_kernels(network, supported_only=True)
    ts, tu, _ = RuntimeAnalyzer.aggregate_kernels_runtime(kernels, Timers.CUEvents.cuda_graphs, 10, False)
    logger.info(f"{ts}, {ts + tu}")


if __name__ == '__main__':
    _bm = "alexnet"
    run(_bm)
