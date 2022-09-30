import logging
import multiprocessing as mp
import subprocess as sp
import time
from collections import defaultdict

import pyutils.hosts.agx as agx
from pyutils.common.paths import WATSSUP_PATH
from pyutils.common.strings import S_P_ALL, S_INTERNAL_PWR_METER, S_EXTERNAL_PWR_METER
from pyutils.common.timers import StopWatch
from pyutils.common.utils import FileUtils
from pyutils.common.utils import GlobalLogger


def collect_pwr_samples(pwr_meter, exit_flag: mp.Event, saved_results: mp.Event,
                        keep_sampling: mp.Event, tmp_results_ready: mp.Event,
                        num_observations: mp.Value,
                        sampling_rate, tmp_file, pausing_tmp_file):
    logger = GlobalLogger(logging.INFO).get_logger()
    res = defaultdict(list)
    stopwatch = StopWatch()
    stopwatch.start()
    pwr_key = 'p_all'

    if pwr_meter == 'external':
        pwr_key = 'p_wattsup'
        with sp.Popen(PowerMonitor.WATTSUP_CMD, stdout=sp.PIPE, bufsize=1, encoding="utf-8",
                      universal_newlines=True) as p:
            for line in p.stdout:
                # Check if we are done measuring power
                if exit_flag.is_set():
                    p.kill()
                    logger.info("Breaking the power sampling loop.")
                    break

                # Check if power sampling is paused for quality verification
                if not keep_sampling.is_set():
                    logger.info("Paused to check observations ...")
                    FileUtils.silent_remove(pausing_tmp_file)  # Make sure we are writing a new file.
                    FileUtils.serialize(res[pwr_key], pausing_tmp_file, save_as='list')  # Write the samples we
                    # collected so far to the tmp file
                    tmp_results_ready.set()  # Notify the calling process that the samples are written and
                    # ready for inspection
                    keep_sampling.wait()  # Wait until the calling process asks for resumption
                    logger.info("Waiting is done.")

                # Read a power measurement from the wattsup meter.
                if line != '\n' and line != ' ':
                    line = float(line.replace("\n", ''))
                    res[pwr_key].append(line)

                # Update the number of observations collected so far.
                num_observations.value = len(res[pwr_key])

    if pwr_meter == 'internal':
        while True:
            if exit_flag.is_set():
                logger.info("Breaking the power sampling loop.")
                break

            # Check if power sampling is paused for quality verification
            if not keep_sampling.is_set():
                logger.info("Paused to check observations ...")
                FileUtils.silent_remove(pausing_tmp_file)  # Make sure we are writing a new file.
                FileUtils.serialize(res[pwr_key], pausing_tmp_file, save_as='list')  # Write the samples we
                # collected so far to the tmp file
                tmp_results_ready.set()  # Notify the calling process that the samples are written and
                # ready for inspection
                keep_sampling.wait()  # Wait until the calling process asks for resumption
                logger.info("Waiting is done.")

            p = agx.PowerMonitor.get_pmu_reading_all_watts()
            # t = agx.ThermalMonitor.get_temp_all()

            res['p_gpu'].append(p["gpu"])
            res['p_cpu'].append(p["cpu"])
            res['p_soc'].append(p["soc"])
            res['p_ddr'].append(p["ddr"])
            res['p_sys'].append(p["sys"])
            # res['p_cv'].append(p["cv"])
            res['p_all'].append(p["all"])
            num_observations.value = len(res[pwr_key])
            time.sleep(1 / sampling_rate)

    res['duration_sec'] = stopwatch.elapsed_s()
    # NOTE: we save to file because queue blocks at certain sizes.
    FileUtils.serialize(res, tmp_file)
    logger.info("Saved results from monitor module")
    saved_results.set()
    logger.info("Run method is completed.")


class PowerMonitor(mp.Process):
    tmp_file = "/tmp/tmp_data.dat"
    pausing_tmp_file = "/tmp/tmp_pausing_data.json"
    WATTSUP_CMD = [WATSSUP_PATH, "ttyUSB0", "watts"]

    def __init__(self, power_meter: str = 'internal', logging_level=logging.INFO,
                 sampling_rate=1):
        self.exit_flag = mp.Event()
        self.saved_results = mp.Event()
        self.keep_sampling = mp.Event()
        self.tmp_results_ready = mp.Event()
        self.keep_sampling.set()
        self.num_observations = mp.Value("i", 0)
        self.power_meter = power_meter

        # clear any tmp files that might exist from previous runs.
        FileUtils.silent_remove(self.tmp_file)
        FileUtils.silent_remove(self.pausing_tmp_file)

        self.readings = None
        super().__init__(target=collect_pwr_samples,
                         args=(power_meter, self.exit_flag, self.saved_results, self.keep_sampling,
                               self.tmp_results_ready, self.num_observations, sampling_rate,
                               self.tmp_file, self.pausing_tmp_file))
        self.logger = GlobalLogger(logging_level).get_logger()

    def resume_sampling(self):
        self.keep_sampling.set()
        self.logger.info("Resume monitor.")

    def pause_sampling(self):
        self.tmp_results_ready.clear()
        self.keep_sampling.clear()
        self.logger.info("Pause monitor.")

    def is_paused(self):
        self.logger.info(f"Is paused? -> {not self.keep_sampling.is_set()}")
        return not self.keep_sampling.is_set()

    def is_stopped(self):
        return self.exit_flag.is_set()

    def get_records(self):
        if self.is_paused():
            self.tmp_results_ready.wait()
            records = FileUtils.deserialize(self.pausing_tmp_file)
            return records
        if self.is_stopped():
            if self.power_meter == S_INTERNAL_PWR_METER:
                return self.readings[S_P_ALL]
            elif self.power_meter == S_EXTERNAL_PWR_METER:
                return self.readings['p_wattsup']
            else:
                raise Exception("Unsupported power meter.")

        return None

    def get_num_observations(self):
        return self.num_observations.value

    def stop(self):
        if self.is_paused():
            self.resume_sampling()
        self.exit_flag.set()
        self.logger.info("Set monitor exit flag.")
        self.join(timeout=10)
        self.logger.info("Monitor process done. Waiting for results to be saved to disk...")
        counter = 0
        while not self.saved_results.is_set() and counter < 600:
            time.sleep(0.1)
            counter += 1
        if counter == 600 and not self.saved_results.is_set():
            raise Exception("Timed out while waiting on saving results to disk.")
        else:
            self.logger.info("Parsing readings and preparing for final exit.")
            self.readings = FileUtils.deserialize(self.tmp_file)
            FileUtils.silent_remove(self.tmp_file)
            FileUtils.silent_remove(self.pausing_tmp_file)
        self.logger.info("Exiting monitor.")
