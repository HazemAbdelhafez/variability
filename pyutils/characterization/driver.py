import argparse
import gc
import os
import sys
import traceback
import warnings
import psutil

from pyutils.characterization.common.utils import parse_measurement_args, parse_parameters, \
    load_module_and_input, calculate_block_size, MAX_BLOCK_SIZE, get_output_file_path
from pyutils.common.config import Timers
from pyutils.common.experimental_data_managers import TorchModuleExpData
from pyutils.common.experiments_utils import Measurements, DVFSManager, TorchSettings
from pyutils.common.methods import join, is_runtime, is_power
from pyutils.common.paths import LOGGING_DIR
from pyutils.common.strings import S_ESTIMATED_BLOCK_SIZE, S_RECORD_DATA, S_TIMER_OVERHEAD, S_METRIC
from pyutils.common.utils import GlobalLogger, FileUtils, TimeStamp
from pyutils.cuda_graphs.utils.timers import ad_hoc_fix_for_cuda_graphs, get_single_runtime_observation, \
    construct_cuda_graph, \
    time_cuda_graph, warmup_cuda_graph
from pyutils.run.config import I_RERUN_EXPERIMENT_CODE

logger_obj = GlobalLogger()
logger_obj.set_out_file(join(LOGGING_DIR, "jobs", "run", f"job_{TimeStamp.get_minute_timestamp()}.log"))
logger = logger_obj.get_logger()

warnings.filterwarnings("ignore")


def get_env_data():
    pid = os.getpid()
    p = psutil.Process(pid)
    env_vars = p.environ()
    mem_mapping = p.memory_maps(grouped=False)
    return {"pid": pid, "environment_variables": env_vars, "memory_maps": mem_mapping}


def characterize_model_time(config: dict):
    args = parse_measurement_args(config)
    logger.info(f"Measuring {args.metric} for:               {args.bm}")

    params = parse_parameters(config)
    logger.info(f"-- Module parameters: {params.to_str()}")

    # Initialize DVFS manager.
    dvfs_manager = DVFSManager(args)
    dvfs_manager.setup()

    # Check if check for previously characterized kernels
    # if already_characterized(args, params.to_str(), dvfs_manager.dvfs_config):
    #     return

    # Initialize experiment data manager.
    experiment_data_manager = TorchModuleExpData()
    experiment_data_manager.parse_args(args)

    # In this code version, we don't run except cuda graphs. So this is a sanity check.
    assert args.timing_method == Timers.CUEvents.cuda_graphs

    # Setup PyTorch settings
    TorchSettings.setup(args)
    TorchSettings.set_optimization_config()

    # Get the module and its input
    scripted_module, module_input = load_module_and_input(params)

    # TODO: If I don't run these two iterations, the CUDA graph capturing fails, I have no time to
    #  investigate it
    ad_hoc_fix_for_cuda_graphs(scripted_module, module_input)

    graph = construct_cuda_graph(args, scripted_module, module_input)
    warmup_cuda_graph(args, graph)

    # Get thermal readings before experiment
    experiment_data_manager.collect_temp('before')

    if args.control_scenario == 5:
        gc.disable()

    # Set the board to the selected DVFS config.
    dvfs_manager.pre_characterization()

    # Save dvfs config to experiment data.
    experiment_data_manager.save_dvfs_config(dvfs_manager.dvfs_config)

    initial_block_size_runtime = get_single_runtime_observation(graph)

    logger.info(f"Initial block size of {args.block_size} runtime is: {initial_block_size_runtime}")
    # We need to make sure that the block size meets certain criteria, so we calculate it here.
    args.block_size, timer_overhead = calculate_block_size(args, initial_block_size_runtime)
    experiment_data_manager.save_extra_config({S_ESTIMATED_BLOCK_SIZE: args.block_size})
    experiment_data_manager.save_extra_config({S_TIMER_OVERHEAD: timer_overhead})

    # Reconstruct the graph if needed
    if args.graph_size != args.block_size:
        args.graph_size = args.block_size
        graph = construct_cuda_graph(args, scripted_module, module_input)

    # Run the characterization with the new block size (if any)
    observations = time_cuda_graph(args, graph)

    # Reset board settings
    dvfs_manager.post_characterization()

    met_criteria, rme = Measurements.collected_enough_measurements(observations, args.rme,
                                                                   args.num_observations,
                                                                   args.confidence_lvl, return_rme=True)

    low_impact_kernel = args.graph_size >= MAX_BLOCK_SIZE

    if low_impact_kernel:
        logger.warning(f"Low impact kernel detected: {args.bm} at DVFS config: {dvfs_manager.dvfs_config}")

    if not met_criteria:
        logger.warning(
            f"DVFS config: {dvfs_manager.dvfs_config}, did not meet quality criteria with RME: {rme}")

    invalid_data = (not met_criteria) and (not low_impact_kernel) and args.check_rme

    if args.control_scenario == 5:
        gc.enable()

    if invalid_data:
        experiment_data_manager.data[S_RECORD_DATA] = False

    # Get thermal readings after experiment
    experiment_data_manager.collect_temp('after')

    # Save collected metric data
    experiment_data_manager.save_metric_data(observations)

    # Save kernel parameters
    experiment_data_manager.save_module_parameters(params)

    # Save extra data
    # env_data = get_env_data()
    # experiment_data_manager.save_extra_config(env_data)

    # Create experiment output data directory
    output_path = get_output_file_path(args)
    experiment_data_manager.write_to_disk(output_path)

    if invalid_data:
        # This will tell the calling process to repeat this experiment.
        sys.exit(I_RERUN_EXPERIMENT_CODE)


# def characterize_model_power(args: NetworksBlockBasedMeasurePowerArgs):
#     logger.info(f"Measuring {args.metric} for:               {args.bm}")
#
#     # Initialize experiment data manager.
#     experiment_data_manager = TorchModuleExpData()
#     experiment_data_manager.parse_args(args)
#
#     # Initialize DVFS manager.
#     dvfs_manager = DVFSManager(args)
#     dvfs_manager.setup()
#
#     # Setup PyTorch settings
#     TorchSettings.setup(args)
#
#     # Get input tensor
#     input_cpu = ModelsAndInputs.get_example_cpu(model_name=args.bm, batch_size=args.batch_size)
#
#     # Load scripted model
#     scripted_model = ModelsAndInputs.get_model(args.bm)
#
#     # Init timer
#     timer = CudaStopWatch()
#
#     # Warm up
#     result_cpu = None
#     logger.info(f"Warmup for {args.warmup_itrs} iterations")
#     for _ in range(args.warmup_itrs):
#         input_gpu = input_cpu.cuda(non_blocking=True)
#         result_gpu = scripted_model(input_gpu)
#         result_cpu = result_gpu.cpu()
#     torch.cuda.synchronize()
#     logger.info("Done warmup")
#
#     # Get thermal readings before experiment
#     experiment_data_manager.collect_temp('before')
#
#     if args.control_scenario == 5:
#         gc.disable()
#
#     # Set the board to the selected DVFS config.
#     dvfs_manager.pre_characterization()
#
#     # Save dvfs config to experiment data.
#     experiment_data_manager.save_dvfs_config(dvfs_manager.dvfs_config)
#
#     input_cpu = ModelsAndInputs.get_example_cpu(model_name=args.bm, batch_size=args.batch_size)
#     overhead = Measurements.cuda_sync_overhead()
#
#     # Measure the initial block size runtime.
#     timer.start()
#     for _ in range(args.block_size):
#         input_gpu = input_cpu.cuda(non_blocking=True)
#         result_gpu = scripted_model.forward(input_gpu)
#         result_cpu = result_gpu.cpu()
#     timer.stop()
#     torch.cuda.synchronize()
#
#     # Estimate the target block size - faster than incrementing and trying again.
#     min_runtime_to_discard_overhead = overhead / args.block_overhead_threshold
#     target_block_runtime = max(min_runtime_to_discard_overhead, args.block_runtime_ms)
#     current_block_runtime = timer.elapsed_ms()
#     new_block_size = int(1.1 * args.block_size * target_block_runtime / current_block_runtime)  # 10%
#     safety factor
#     logger.info(f"Current block runtime: {current_block_runtime}, Target block runtime: {
#     target_block_runtime}")
#
#     args.block_size = max(new_block_size, args.block_size)
#     experiment_data_manager.save_extra_config({S_ESTIMATED_BLOCK_SIZE: args.block_size})
#     logger.info("Estimated block size: %d" % args.block_size)
#
#     monitor = PowerMonitor(args.pwr_meter, sampling_rate=args.sampling_rate)
#     logger.info(f"Initialized Monitor module with {args.pwr_meter} power meter.")
#
#     monitor.start()
#
#     # Run the inference task.
#     for _ in range(args.block_size):
#         input_gpu = input_cpu.cuda(non_blocking=True)
#         result_gpu = scripted_model.forward(input_gpu)
#         result_cpu = result_gpu.cpu()
#
#     monitor.stop()
#     n = monitor.get_num_observations()
#     full_records = monitor.get_records()
#     pwr_observations = full_records[FRONT_IGNORED_POWER_RECORDS:-TAIL_IGNORED_POWER_RECORDS]
#
#     # Reset board settings
#     dvfs_manager.post_characterization()
#     torch.cuda.synchronize()
#     met_criteria = Measurements.collected_enough_measurements(pwr_observations, rme_threshold=args.rme,
#                                                               confidence_lvl=args.confidence_lvl)
#
#     if not met_criteria:
#         logger.info(f"DVFS config: {dvfs_manager.dvfs_config_idx}, did not met criteria with RME")
#
#     invalid_data = n < args.num_observations or not met_criteria
#     invalid_data = invalid_data and experiment_data_manager.data[S_RECORD_DATA]
#
#     if args.control_scenario == 5:
#         gc.enable()
#
#     # Get thermal readings after experiment
#     experiment_data_manager.collect_temp('after')
#
#     if invalid_data:
#         experiment_data_manager.data[S_RECORD_DATA] = False
#
#     # Save collected metric data
#     experiment_data_manager.save_metric_data(copy(monitor.readings))
#
#     output_path = \
#         prepare(CHARACTERIZATION_DATA_DIR, HOSTNAME, args.bm, args.metric,
#                 f"raw_data_{HOUR_TIMESTAMP}.json")
#     experiment_data_manager.write_to_disk(output_path)
#
#     if invalid_data:
#         # This will tell the calling process to repeat this experiment.
#         sys.exit(I_RERUN_EXPERIMENT_CODE)


def entry(args):
    config = FileUtils.deserialize(args['config_file'])

    if is_runtime(config[S_METRIC]):
        characterize_model_time(config)

    elif is_power(config[S_METRIC]):
        pass

    else:
        print("Invalid input parameters.")


if __name__ == '__main__':
    try:

        # Init env
        os.environ["CUSTOM_PROFILING"] = "0"
        os.environ["CUSTOM_WORLD_TIMER"] = "0"

        args_parser = argparse.ArgumentParser()
        args_parser.add_argument("-c", "--config_file", required=True, help="")

        cmd_args = vars(args_parser.parse_args())

        entry(cmd_args)
    except Exception as e:
        st_trace = str(traceback.format_exc())
        msg = str(e) + "\n" + st_trace + "\n"
        print(e)
        print(st_trace)
        sys.exit(1)
