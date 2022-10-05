import argparse
import copy
import os
import subprocess as sp
import sys
import traceback
from os.path import join as jp


from pyutils.run.utils import Logging, EnvHandler

try:
    from pyutils.characterization.kernels.utils.checks import is_kernel
    from pyutils.hosts.agx import DVFSHelpers
    from pyutils.common import slack_notifier
    from pyutils.common.strings import S_KERNEL_PARAMETERS, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ, \
    S_NUM_OBSERVATIONS, S_BLOCK_SIZE, S_BLOCK_RUNTIME_MS, S_CHECK_RME, S_GRAPH_SIZE
    from pyutils.common.timers import StopWatch
    from pyutils.common.strings import S_BENCHMARK, S_RECORD_DATA, S_DVFS_CONFIG_IDX, \
        S_DVFS_CONFIG_FILE_PATH, S_NUM_RESTARTS, S_RESTART_ID, S_DISCARDED_RESTARTS, S_EXP_TIMEOUT_DURATION, \
        S_MAX_NUM_RETRIES
    from pyutils.common.paths import JOBS_CFG_DIR, CHARACTERIZATION_DATA_DIR, LOGGING_DIR
    from pyutils.run.utils import PathsHandler, HostsHandler
    from pyutils.run.config import S_DVFS_CONFIGS, I_RERUN_EXPERIMENT_CODE
    from pyutils.common.utils import GlobalLogger, FileUtils, TimeStamp
    from pyutils.characterization.networks.analyzers import stack_trace

    if os.path.exists(jp(f'{os.uname().nodename}_job_failed.txt')):
        os.remove(jp(f'{os.uname().nodename}_job_failed.txt'))

    remote_status_dir = PathsHandler.get_remote_status_dir()
    remote_python_executable_path = PathsHandler.get_python_remote_path()
    remote_host_name = HostsHandler.get_remote_logger_hostname()

    local_status_dir = PathsHandler.get_local_status_dir()
    local_host_name = HostsHandler.get_local_hostname()
    local_job_log_dir = PathsHandler.get_local_logging_dir()

    logger_obj = GlobalLogger()
    logger_obj.set_out_file(jp(LOGGING_DIR, "jobs", "launch", f"job_{TimeStamp.get_minute_timestamp()}.log"))
    logger = logger_obj.get_logger()

except ImportError as import_err:
    before_run_error_status_file = jp(f'{os.uname().nodename}_job_failed.txt')
    before_run_err_msg = str(import_err) + "\n"
    before_run_err_msg += traceback.format_exc() + "\n"
    Logging.write_and_send(before_run_err_msg, file_path=before_run_error_status_file, append=False)
    sys.exit(1)

except Exception as general_err:
    before_run_error_status_file = jp(f'{os.uname().nodename}_job_failed.txt')
    before_run_err_msg = str(general_err) + "\n"
    before_run_err_msg += traceback.format_exc() + "\n"
    Logging.write_and_send(before_run_err_msg, file_path=before_run_error_status_file, append=False)
    sys.exit(1)


def get_ts():
    return TimeStamp.get_timestamp('%d/%m/%Y %H:%M:%S')


class Driver:
    @staticmethod
    def run_bm(cmd, timeout):
        cmd = [str(i) for i in cmd]
        return sp.run(cmd, capture_output=True, text=True, env=os.environ, timeout=timeout)

    @staticmethod
    def handle_success(status: sp.CompletedProcess):
        output_log_file = jp(local_job_log_dir, f'{local_host_name}_{TimeStamp.get_timestamp()}.txt')
        output_msg = status.stdout + "\n" + status.stderr + "\n"
        Logging.write_locally(output_msg, output_log_file)

    @staticmethod
    def handle_failure(status: sp.CompletedProcess, bm, cmd: list, optional_msg: str = ""):
        error_status_file = jp(local_status_dir, f'{local_host_name}_error_{bm}.txt')
        err_msg = "" if optional_msg == "" else optional_msg
        err_msg += f"Failed command: {' '.join(cmd)}\n"
        err_msg += status.stdout + '\n' + '-----\n' + status.stderr
        Logging.write_and_send(err_msg, file_path=error_status_file, append=True)


# Progress reporting threshold map
def get_progress_report_threshold(num_runs):
    num_runs = int(num_runs)
    if num_runs >= 1000:
        threshold = 500
    elif 1000 > num_runs >= 100:
        threshold = 100
    elif 100 > num_runs >= 50:
        threshold = 25
    elif 50 > num_runs >= 5:
        threshold = 5
    else:
        threshold = 1
    return threshold


def board_warmup(cmd, cfg: dict, tmp_config_file: str):
    dvfs_config_index = 0
    bm = cfg[S_BENCHMARK]
    local_status_file = jp(local_status_dir, f'{local_host_name}_started_{bm}.txt')

    # Run a dummy characterization few times before collecting any characterization data to reach steady state.
    num_discarded_restarts = int(cfg[S_DISCARDED_RESTARTS])

    tmp_config = copy.copy(cfg)
    tmp_config.update({S_RECORD_DATA: False})
    tmp_config.update({S_DVFS_CONFIG_IDX: dvfs_config_index})
    tmp_config.update({S_RESTART_ID: -1})
    tmp_config.update({S_DVFS_CONFIG_FILE_PATH: 'dvfs_max.json'})
    tmp_config.update({S_NUM_OBSERVATIONS: 31})
    tmp_config.update({S_BLOCK_SIZE: 31})
    tmp_config.update({S_GRAPH_SIZE: 31})
    tmp_config.update({S_BLOCK_RUNTIME_MS: 500})
    tmp_config.update({S_CHECK_RME: False})
    FileUtils.serialize(tmp_config, file_path=tmp_config_file, append=False)

    for i in range(num_discarded_restarts):

        msg = f"[DISCARDED RESTARTS] Warming up '{bm}' at DVFS index: {dvfs_config_index} | restart id {i}"
        Logging.write_and_send(msg, local_status_file)

        try:
            # Execute the profiling command
            status = Driver.run_bm(cmd, int(cfg[S_EXP_TIMEOUT_DURATION]))

            # Success
            if status.returncode == 0:
                Driver.handle_success(status)

            elif status.returncode == I_RERUN_EXPERIMENT_CODE:
                # Ignore in warmup
                pass

            else:  # Unhandled failure
                err_msg = f"{bm} failed during warmup at dvfs config {dvfs_config_index}"
                Driver.handle_failure(status, bm, cmd, err_msg)

        except sp.TimeoutExpired as _:
            warning_status_file = jp(local_status_dir, f'{local_host_name}_warning_{bm}.txt')
            warning_msg = f"Process timed out during warmup for {bm} at dvfs config {dvfs_config_index}."
            Logging.write_and_send(warning_msg, file_path=warning_status_file, append=True)

        except sp.CalledProcessError as sp_e:
            error_status_file = jp(local_status_dir, f'{local_host_name}_error_{bm}.txt')
            err_msg = str(sp_e)
            Logging.write_and_send(err_msg, file_path=error_status_file)
            return


def benchmark_characterization(cfg, report_end=True):
    # Parse job configuration
    bm = cfg[S_BENCHMARK]
    dvfs_config_indexes = cfg[S_DVFS_CONFIGS]
    max_num_retries = int(cfg[S_MAX_NUM_RETRIES]) - 1  # Repeats tracker is zero-indexed
    threshold = get_progress_report_threshold(len(dvfs_config_indexes) * int(cfg[S_NUM_RESTARTS]))

    logger.info(f"Job config: {cfg}")

    # Maximum number of repeats before reporting weird behavior is 3
    repeats_tracker = dict()

    tmp_config_file = "/tmp/characterization_job_cfg.json"
    local_status_file = jp(local_status_dir, f'{local_host_name}_started_{bm}.txt')
    warning_status_file = jp(local_status_dir, f'{local_host_name}_warning_{bm}.txt')

    msg = f'Started {bm} characterization experiment on {local_host_name}\n'
    Logging.write_and_send(msg, file_path=local_status_file, append=False)

    msg = f'-- Setting environment on {local_host_name}'
    Logging.write_locally(msg, local_status_file)
    EnvHandler.set_env(cfg)

    # Set benchmark specific settings
    cmd = [remote_python_executable_path, '-m', 'pyutils.characterization.driver', '-c', tmp_config_file]

    msg = f'-- Warming up on {local_host_name}'
    Logging.write_and_send(msg, local_status_file)
    board_warmup(cmd, cfg, tmp_config_file)

    loop_timer = StopWatch()

    dvfs_idx = 0
    if HostsHandler.get_local_hostname() == "node-05":
        dvfs_idx = 216
    elif HostsHandler.get_local_hostname() == "node-12":
        dvfs_idx = 272
    else:
        dvfs_idx = 0
    tmp_config = copy.copy(cfg)
    Logging.write_and_send("Staring profiling iterations.", local_status_file)
    while dvfs_idx < len(dvfs_config_indexes):
        restart_idx = 0
        dvfs_config_index = dvfs_config_indexes[dvfs_idx]
        tmp_config.update({S_DVFS_CONFIG_IDX: dvfs_config_index})

        msg = f"Profiling {bm} at DVFS config: {dvfs_config_index}"
        Logging.write_locally(msg, local_status_file)

        while restart_idx < int(cfg[S_NUM_RESTARTS]):
            # Check that this DVFS config did not timeout in a previous restart and that we decided to ignore it
            if repeats_tracker.get(dvfs_config_index, 0) >= max_num_retries:
                warning_msg = f"Ignoring remaining restarts of DVFS config: {dvfs_idx} " \
                              f"due to exceeding maximum number of retries in restart: {restart_idx + 1}"
                Logging.write_and_send(warning_msg, file_path=warning_status_file, append=True)
                break

            tmp_config.update({S_RESTART_ID: restart_idx})
            FileUtils.serialize(tmp_config, file_path=tmp_config_file, append=False)

            loop_timer.start()
            try:
                # Execute the profiling command
                status = Driver.run_bm(cmd, int(cfg[S_EXP_TIMEOUT_DURATION]))
                # Success
                if status.returncode == 0:
                    Driver.handle_success(status)

                elif status.returncode == I_RERUN_EXPERIMENT_CODE:
                    # In some cases we might want to repeat an experiment, this is where we do that.
                    num_repeats = repeats_tracker.get(dvfs_config_index, 0)
                    if num_repeats >= max_num_retries:
                        warning_msg = f"-- Repeated {num_repeats} times " \
                                      f"at dvfs config {dvfs_config_index} - restart {restart_idx}."
                        Logging.write_and_send(warning_msg, file_path=warning_status_file, append=True)
                    else:
                        # Do not increment the index
                        msg = f"-- Repeating DVFS config {dvfs_config_index} - restart {restart_idx}"
                        Logging.write_and_send(msg, local_status_file, append=True)
                        repeats_tracker[dvfs_config_index] = repeats_tracker.get(dvfs_config_index, 0) + 1
                        continue

                else:  # Unhandled failure
                    err_msg = f"-- Characterizing {bm} failed at dvfs config {dvfs_config_index}"
                    Driver.handle_failure(status, bm, cmd, err_msg)

            except sp.CalledProcessError as sp_e:
                error_status_file = jp(local_status_dir, f'{local_host_name}_error_{bm}.txt')
                err_msg = str(sp_e)
                Logging.write_and_send(err_msg, file_path=error_status_file)
                return -1

            except sp.TimeoutExpired as _:
                # This exception is raised after the child process receives a kill signal.
                # Just retry it
                num_repeats = repeats_tracker.get(dvfs_config_index, 0)
                if num_repeats >= max_num_retries:
                    warning_msg = f"-- Process timed out and characterizing {bm} repeated {num_repeats} times " \
                                  f"at dvfs config {dvfs_config_index} - restart {restart_idx}."
                    Logging.write_and_send(warning_msg, file_path=warning_status_file, append=True)
                else:
                    msg = f"-- Process timed out, repeating config: {dvfs_config_index}. " \
                          f"Currently at repeat no. {repeats_tracker.get(dvfs_config_index, 0)} - restart {restart_idx}"
                    Logging.write_and_send(msg, warning_status_file, append=True)
                    repeats_tracker[dvfs_config_index] = repeats_tracker.get(dvfs_config_index, 0) + 1
                    continue

            # Progress report
            curr_run = int(dvfs_idx * int(cfg[S_NUM_RESTARTS]) + restart_idx)
            msg = f'-- Done DVFS config: {dvfs_idx} - restart {restart_idx} in {loop_timer.elapsed_s()} sec.'
            Logging.write_locally(msg, local_status_file)

            if curr_run % threshold == 0 or restart_idx == int(cfg[S_NUM_RESTARTS]) - 1:
                Logging.send(local_status_file)

            restart_idx += 1

        dvfs_idx += 1

    if report_end:
        # End of experiment report
        end_status_file = jp(local_status_dir, f'{local_host_name}_done.txt')
        msg = f'Done characterization experiment for {bm} on {local_host_name}'
        Logging.write_and_send(msg, file_path=end_status_file, append=False)
        slack_notifier.send(msg)
    return 0


def benchmark_kernels_characterization(cfg):
    kernels = stack_trace.get_unique_kernels(cfg[S_BENCHMARK], supported_only=False)
    for kernel_name in kernels.keys():
        cfg[S_BENCHMARK] = kernel_name
        for kernel_params in kernels.get(kernel_name):
            cfg[S_KERNEL_PARAMETERS] = kernel_params.to_dict()
            status = benchmark_characterization(cfg, report_end=False)
            if status != 0:
                return

    # End of experiment report
    end_status_file = jp(local_status_dir, f'{local_host_name}_done.txt')
    msg = f'Done characterization experiment for {cfg[S_BENCHMARK]} on {local_host_name}'
    Logging.write_and_send(msg, file_path=end_status_file, append=False)
    slack_notifier.send(msg)


def entry():
    job_cfg = ""
    try:
        argument_parser = argparse.ArgumentParser()
        argument_parser.add_argument("-m", "--mode", required=True)
        argument_parser.add_argument("-c", "--config_file", required=True)

        cmd_args = vars(argument_parser.parse_args())
        mode = cmd_args['mode']
        job_cfg_file = cmd_args['config_file']
        job_cfg = FileUtils.deserialize(jp(JOBS_CFG_DIR, job_cfg_file))

        if mode == 'benchmark_characterization':
            benchmark_characterization(job_cfg)
        elif mode == "benchmark_kernels_characterization":
            benchmark_kernels_characterization(job_cfg)
        else:
            raise Exception("Unsupported mode run mode")

    except (Exception, SystemExit) as e:
        error_status_file = jp(local_status_dir, f'{local_host_name}_job_failed.txt')
        err_msg = str(e) + "\n"
        err_msg += traceback.format_exc() + "\n"
        err_msg += "Job config: \n" + str(job_cfg) + "\n"
        Logging.write_and_send(err_msg, file_path=error_status_file, append=False)


if __name__ == '__main__':
    entry()
