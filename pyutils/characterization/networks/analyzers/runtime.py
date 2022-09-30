import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from pyutils.characterization.kernels.utils.checks import KernelsChecks
from pyutils.characterization.kernels.utils.loaders import KernelsLoaders
from pyutils.characterization.kernels.utils.performance import KernelRuntime
from pyutils.common.arguments import (KernelMeasureTimeArgs)
from pyutils.common.config import Timers
from pyutils.common.experiments_utils import TorchSettings
from pyutils.common.methods import join as jp
from pyutils.common.methods import json_convert
from pyutils.common.paths import TMP_DIR
from pyutils.common.timers import CudaStopWatch
from pyutils.common.utils import GlobalLogger
from pyutils.cuda_graphs.utils.timers import collect_time_observations
from pyutils.hosts.agx import DVFSHelpers
from pyutils.run.utils import PathsHandler

logger = GlobalLogger().get_logger()


class RuntimeAnalyzer:
    """ This class is used to extract the top-level kernels (conv, batchnorm2d, matmul, etc.), and time them."""

    @staticmethod
    def _time_kernel_with_nvprof(kernel_name, module_params_dict, iterations):
        os.makedirs(TMP_DIR, exist_ok=True)
        tmp_params_file = jp(TMP_DIR, f'{kernel_name}_tmp_params.txt')
        tmp_prof_file = jp(TMP_DIR, f'{kernel_name}_tmp_output.prof')
        with open(tmp_params_file, 'w') as params_file:
            json.dump(module_params_dict, params_file, default=json_convert)
        cmd = f"sudo /usr/local/cuda/bin/nvprof --quiet --unified-memory-profiling off --profile-from-start off " \
              f"-f -o {tmp_prof_file} {PathsHandler.get_python_remote_path()} " \
              f"-m pyutils.profile_module -k {kernel_name} -p {tmp_params_file} -t {iterations}"

        os.system(cmd)
        os.remove(tmp_params_file)
        # Parse profiler output
        output = torch.autograd.profiler.load_nvprof(tmp_prof_file)
        sum_t = 0
        count = 0  # For some kernels they are more than once per iteration, so we keep the count anyway
        for row in output:
            reported_kernel_name = row.name.split(',')[0]
            # Special ops names cases
            valid_kernel = (reported_kernel_name == kernel_name or reported_kernel_name == 'FusionGroup')
            valid_kernel = valid_kernel or (
                    KernelsChecks.is_add(reported_kernel_name) and KernelsChecks.is_add(kernel_name))
            valid_kernel = valid_kernel or (
                    KernelsChecks.is_mm(reported_kernel_name) and KernelsChecks.is_mm(kernel_name))
            valid_kernel = valid_kernel or (
                    KernelsChecks.is_bn(reported_kernel_name) and KernelsChecks.is_bn(kernel_name))
            valid_kernel = valid_kernel or (
                    KernelsChecks.is_maxpool(reported_kernel_name) and KernelsChecks.is_maxpool(kernel_name))
            valid_kernel = valid_kernel or (
                    KernelsChecks.is_relu(reported_kernel_name) and KernelsChecks.is_relu(kernel_name))
            if valid_kernel:
                sum_t += row.cuda_time / 1e6
                count += 1

        os.remove(tmp_prof_file)
        return [sum_t / count]

    @staticmethod
    def _time_kernel_with_block(module, input_obj, timing_itrs):
        results = [0.0 for _ in range(timing_itrs)]
        timer = CudaStopWatch()
        block_size = 100
        for i in range(timing_itrs):
            timer.start()
            for _ in range(block_size):
                _ = module.forward(input_obj)
            timer.stop()
            torch.cuda.synchronize()
            results[i] = timer.elapsed_ms(block_size)
        return results

    @staticmethod
    def _time_kernel_with_cuda_graphs(module, input_obj, args: KernelMeasureTimeArgs):
        elapsed_time_ms = collect_time_observations(args, module, input_obj)
        return elapsed_time_ms

    @staticmethod
    def _time_kernel_with_per_run(module, input_obj, timing_itrs):
        results = [0.0 for _ in range(timing_itrs)]
        _input_obj = torch.rand(1, 3, 512, 512).cuda()
        conv = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2).cuda().eval()
        conv = torch.jit.script(conv)
        timer = CudaStopWatch()
        for i in range(timing_itrs):
            _ = conv(_input_obj)
            timer.start()
            _ = module.forward(input_obj)
            timer.stop()
            torch.cuda.synchronize()
            results[i] = timer.elapsed_ms()
        return results

    @staticmethod
    def _time_kernel_overall(module, input_obj, timing_itrs):
        timer = CudaStopWatch()
        timer.start()
        timer.start_event.synchronize()
        for i in range(timing_itrs):
            _ = module.forward(input_obj)
        timer.stop()
        timer.stop_event.synchronize()
        torch.cuda.synchronize()
        results = [timer.elapsed_ms() / timing_itrs]
        return results

    @staticmethod
    def time_a_kernel(kernel_name, kernel_parameters, args: KernelMeasureTimeArgs):
        """
        Time a kernel.
        Args:
            kernel_name: str, required
            kernel_parameters: BaseParameters, required
            args: KernelMeasureTimeArgs, required

        Returns:
            results: List[KernelRuntime]

        """
        if KernelsChecks.is_supported(kernel_name):
            if isinstance(kernel_parameters, dict):
                parameters_parser = KernelsLoaders.load_kernel_parameters_parser(kernel_name)
                kernel_parameters = parameters_parser(kernel_parameters)

            if args.timing_method == 'nvprof':
                return RuntimeAnalyzer._time_kernel_with_nvprof(kernel_name, kernel_parameters.to_dict(),
                                                                args.num_observations)

        TorchSettings.set_optimization_config()

        module, input_obj = KernelsLoaders.load_module_and_input(kernel_parameters)
        if module is None or input_obj is None:
            return []

        if args.timing_method != Timers.CUEvents.cuda_graphs:
            for i in range(args.warmup_itrs):
                _ = module.forward(input_obj)
            torch.cuda.synchronize()

        if args.timing_method == Timers.CUEvents.per_run:
            results = RuntimeAnalyzer._time_kernel_with_per_run(module, input_obj, args.num_observations)

        elif args.timing_method == Timers.CUEvents.overall:
            results = RuntimeAnalyzer._time_kernel_overall(module, input_obj, args.num_observations)

        elif args.timing_method == Timers.CUEvents.block_based:
            results = RuntimeAnalyzer._time_kernel_with_block(module, input_obj, args.num_observations)

        elif args.timing_method == Timers.CUEvents.cuda_graphs:
            results = RuntimeAnalyzer._time_kernel_with_cuda_graphs(module, input_obj, args)
        else:
            raise Exception(f"Unknown timing method: {args.timing_method}")

        return results

    @staticmethod
    def time_a_kernel_with_dvfs(kernel_name, kernel_parameters, args: KernelMeasureTimeArgs,
                                dvfs_config: dict):
        """
        Time a kernel but set DVFS configuration before and after.
        Args:
            kernel_name: str, required
            kernel_parameters: BaseParameters, required
            args: KernelMeasureTimeArgs, required
            dvfs_config: Dict, required

        Returns:
            results: List[KernelRuntime]

        """
        DVFSHelpers.set_dvfs_config(dvfs_config)
        results = RuntimeAnalyzer.time_a_kernel(kernel_name, kernel_parameters, args)
        DVFSHelpers.reset_dvfs_settings()
        return results

    @staticmethod
    def aggregate_kernels_runtime(kernels: dict, args: KernelMeasureTimeArgs, show_details=False) \
            -> Tuple[float, float]:
        """
        This method calculates the aggregate runtime of supported and unsupported kernels.
        Args:
            kernels: dict, required
                A dict containing the kernels as keys, and their runtimes combined as a single list value.

            args: KernelMeasureTimeArgs, required
                A kernel time measurement argument object specifying the timing parameters.
            show_details: bool, optional
                Print fine details about the kernels timing or not (default: False).

        Returns:
            supported_overall_time: float
                Aggregate runtime of all supported kernels.
            unsupported_overall_time: float
                Aggregate runtime of all unsupported kernels.
        """
        times = list()

        supported_overall_time = 0
        unsupported_overall_time = 0
        for kernel_name in kernels.keys():
            for kernel_param in kernels.get(kernel_name):
                runtime_ms = RuntimeAnalyzer.time_a_kernel(kernel_name, kernel_param, args)
                runtime_summary = np.median(runtime_ms)
                if KernelsChecks.is_supported(kernel_name):
                    supported_overall_time += runtime_summary
                elif KernelsChecks.is_unsupported(kernel_name):
                    unsupported_overall_time += runtime_summary
                else:
                    raise Exception("Unhandled kernel. Exiting")

                if show_details:
                    print(kernel_name, " : ", runtime_summary)

                times.append({kernel_name: runtime_summary})
        if show_details:
            print(f"Stack and top-level ops runtime: {supported_overall_time, unsupported_overall_time} = "
                  f"{supported_overall_time + unsupported_overall_time}")
        return supported_overall_time, unsupported_overall_time

    @staticmethod
    def time_multiple_kernels(kernels: dict, args: KernelMeasureTimeArgs) -> List[KernelRuntime]:
        result = list()
        for kernel_name in kernels.keys():
            for kernel_param in kernels.get(kernel_name):
                runtime_ms = RuntimeAnalyzer.time_a_kernel(kernel_name, kernel_param, args)
                kernel_performance_summary = KernelRuntime(kernel_name, kernel_name)
                kernel_performance_summary.store_runtime_data(runtime_ms)
                # TODO: save meta data
                result.append(kernel_performance_summary)
        return result
