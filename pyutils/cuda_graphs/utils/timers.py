from typing import List

import torch
from torch import Tensor
from torch.cuda import CUDAGraph
from torch.nn import Module

from pyutils.common.arguments import TorchModuleMeasureTimeArgs
from pyutils.common.timers import CudaStopWatch
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()


def ad_hoc_fix_for_cuda_graphs(module: Module, in_t):
    # TODO: If I don't run these two iterations, the CUDA graph capturing fails, I have no time to investigate it
    for _ in range(2):
        _ = module.forward(in_t)
    torch.cuda.current_stream().synchronize()


def construct_cuda_graph(args: TorchModuleMeasureTimeArgs, module: Module, in_t):
    # Construct timing graph for a given module
    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(args.graph_size):
                _ = module.forward(in_t)
        torch.cuda.current_stream().synchronize()
    torch.cuda.current_stream().wait_stream(s)
    return g


def warmup_cuda_graph(args: TorchModuleMeasureTimeArgs, g: CUDAGraph):
    # Run warmup iterations for a CUDA graph

    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        # Init timer
        timer = CudaStopWatch()

        # Warmup iterations
        timer.start()
        for _ in range(args.warmup_itrs):
            g.replay()
        timer.stop()
        torch.cuda.current_stream().synchronize()
        warmup_time_ms = timer.elapsed_ms()

    torch.cuda.current_stream().wait_stream(s)
    logger.info(f"Done warmup in {warmup_time_ms} (ms)")


def time_cuda_graph(args: TorchModuleMeasureTimeArgs, g: CUDAGraph):
    # Collect timing measurements for a CUDA graph
    elapsed_time_ms = [0.0 for _ in range(args.num_observations)]
    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        timer = CudaStopWatch()
        for i in range(args.num_observations):
            timer.start()
            g.replay()
            timer.stop()
            torch.cuda.current_stream().synchronize()
            elapsed_time_ms[i] = timer.elapsed_ms(args.graph_size)

    return elapsed_time_ms


def get_single_runtime_observation(g: CUDAGraph):
    # Collect a single timing measurement for a CUDA graph
    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        timer = CudaStopWatch()
        timer.start()
        # A graph replaced a block.
        g.replay()  # Graph size is the block size we want to estimate
        timer.stop()
        torch.cuda.current_stream().synchronize()
        elapsed_time_ms = timer.elapsed_ms()
    return elapsed_time_ms


def collect_time_observations(args: TorchModuleMeasureTimeArgs, module: Module, in_t) -> List[float]:
    """
    This method measures the time of a given torch module object that is run using CUDA graphs API.

    Parameters
    ----------
    @config : TimingConfig, required
        An input object specifying timing details.
    @module : Module, required
        The torch module to be timed.
    @in_t : Tensor, required
        Input tensor to use in the module forward (i.e., inference) call.
    @warmup : bool, optional
        Whether to run warmup iterations, or skip this step (default is True).

    Returns
    -------
    elapsed_time_ms : float
        A list of average runtimes of multiple replays of CUDA graph representation of the module.
    out_t : Tensor
        Output tensor from the forward method call.
    """

    ad_hoc_fix_for_cuda_graphs(module, in_t)
    graph = construct_cuda_graph(args, module, in_t)
    warmup_cuda_graph(args, graph)
    return time_cuda_graph(args, graph)


def test():
    params = [[1, 256, 6, 6], 1, -1]
    in_t = torch.randn(size=params[0], device="cuda")
    graph_size = 300
    kernel = torch.flatten
    num_observations = 35

    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(graph_size):
                _ = kernel(in_t, params[1], params[2])
        torch.cuda.current_stream().synchronize()
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.stream(s):
        # Init timer
        timer = CudaStopWatch()

        # Warmup iterations
        timer.start()
        for _ in range(5):
            g.replay()
        timer.stop()
        torch.cuda.current_stream().synchronize()
        warmup_time_ms = timer.elapsed_ms()

    torch.cuda.current_stream().wait_stream(s)
    logger.info(f"Done warmup in {warmup_time_ms} (ms)")

    elapsed_time_ms = [0.0 for _ in range(num_observations)]
    s = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(s):
        timer = CudaStopWatch()
        for i in range(num_observations):
            timer.start()
            g.replay()
            timer.stop()
            torch.cuda.current_stream().synchronize()
            elapsed_time_ms[i] = timer.elapsed_ms(graph_size)
    torch.cuda.current_stream().wait_stream(s)
    print(elapsed_time_ms)


if __name__ == '__main__':
    test()
