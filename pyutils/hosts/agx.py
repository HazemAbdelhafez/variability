import glob
import json
import os
import platform
import subprocess as sp
from random import choice as pick

import pandas as pd

from pyutils.common.paths import DVFS_CONFIGS_DIR
from pyutils.common.strings import S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ
from pyutils.common.utils import GlobalLogger

JP5_RELEASE = "5.10.104-tegra"

# assert platform.release() == JP5_RELEASE

logger = GlobalLogger().get_logger()

AGX_CPU_CORE_COUNT = 8
PRESET_POWER_MODES = 8
DVFS_SPACE_SIZE = 3654


class Defaults:
    CPU_FREQUENCIES = [115200, 192000, 268800, 345600, 422400, 499200, 576000, 652800, 729600, 806400, 883200,
                       960000, 1036800, 1113600, 1190400, 1267200, 1344000, 1420800, 1497600, 1574400, 1651200,
                       1728000, 1804800, 1881600, 1958400, 2035200, 2112000, 2188800, 2265600]
    GPU_FREQUENCIES = [114750000, 216750000, 318750000, 420750000, 522750000, 624750000, 675750000, 828750000,
                       905250000, 1032750000, 1198500000, 1236750000, 1338750000, 1377000000]
    MEMORY_FREQUENCIES = [204000000, 408000000, 665600000, 800000000, 1065600000, 1331200000, 1600000000,
                          1866000000, 2133000000]


class BoardController:

    @staticmethod
    def get_power_mode():
        """ Return current set power model of the board. """
        return Utils.read_subprocess_output(["nvpmodel", "-q"])

    @staticmethod
    def set_power_mode(index):
        if PRESET_POWER_MODES >= index >= 0:
            sp.run(["nvpmodel", "-m %d" % int(index)])
        else:
            print(f"[Error] Invalid power mode {index}. Ignoring this command.")

    @staticmethod
    def reset_board_to_maximum():
        """ Reset the board settings to maximum performance. """
        BoardController.set_power_mode(8)
        CPUController.set_enabled_cores(AGX_CPU_CORE_COUNT)
        # sp.run(["jetson_clocks.sh"])

    @staticmethod
    def get_num_dvfs_configurations():
        num_freqs = CPUController.get_num_freqs() * GPUController.get_num_freqs() * MemoryController.get_num_freqs()
        return num_freqs

    @staticmethod
    def reset_dvfs_settings():
        if platform.processor() != 'aarch64':
            return
        # Reset board to maximum settings
        BoardController.reset_board_to_maximum()
        CPUController.disable_railgate()
        GPUController.disable_railgate()
        FanController.set_speed_to_max()

        # Set both the GPU and CPU scaling governers to userspace
        CPUController.set_scaling_governor('userspace')
        GPUController.set_scaling_governor('userspace')

        # Set lower and upper bounds just to be sure
        CPUController.set_freq_lower_bound(CPUController.get_min_freq())
        CPUController.set_freq_upper_bound(CPUController.get_max_freq())
        GPUController.set_freq_lower_bound(GPUController.get_min_freq())
        GPUController.set_freq_upper_bound(GPUController.get_max_freq())

        # Set current frequency to max
        CPUController.set_freq_to_max()
        GPUController.set_freq_to_max()
        MemoryController.set_freq_to_max()

    @staticmethod
    def set_board_to_mid_settings():
        BoardController.reset_dvfs_settings()
        # Set current frequency to mid
        mid_cpu_freq_idx = int(len(CPUController.get_avail_freq()) / 2)
        mid_gpu_freq_idx = int(len(GPUController.get_avail_freq()) / 2)
        mid_mem_freq_idx = int(len(MemoryController.get_avail_freq()) / 2)

        CPUController.set_freq(CPUController.get_avail_freq()[mid_cpu_freq_idx])
        GPUController.set_freq(GPUController.get_avail_freq()[mid_gpu_freq_idx])
        MemoryController.set_freq(MemoryController.get_avail_freq()[mid_mem_freq_idx])

    @staticmethod
    def set_board_to_lowest_settings():
        BoardController.reset_dvfs_settings()
        CPUController.set_freq(CPUController.get_avail_freq()[0])
        GPUController.set_freq(GPUController.get_avail_freq()[0])
        MemoryController.set_freq(MemoryController.get_avail_freq()[0])


class Utils:
    def __int__(self):
        pass

    @staticmethod
    def write_file(file_path, out):
        """ Write a value to a file. """
        with open(file_path, "w") as f:
            f.write(out)
            f.flush()

    @staticmethod
    def read_dir(dir_path):
        return os.listdir(dir_path)

    @staticmethod
    def read_first_line_from_file(file_path):
        with open(file_path, "r") as f:
            line = f.readline()
        return line.strip()

    @staticmethod
    def read_file(file_path):
        lines = list()
        with open(file_path, "r") as f:
            content = f.readlines()
            for l in content:
                if str(file_path).__contains__('json'):
                    lines.append(json.loads(l))
                else:
                    lines.append(str(l))
        return lines

    @staticmethod
    def read_file_as_int(file_path):
        return int(Utils.read_first_line_from_file(file_path))

    @staticmethod
    def read_file_as_float(file_path):
        return float(Utils.read_first_line_from_file(file_path))

    @staticmethod
    def read_subprocess_output(process_args):
        return sp.check_output(process_args, encoding="utf-8").strip()

    @staticmethod
    def read_int_array_and_sort(file_path):
        line = Utils.read_first_line_from_file(file_path)
        freqs = [int(i) for i in line.split()]
        freqs.sort()
        return freqs

    @staticmethod
    def get_ratio_of_frequencies(freqs, ratio=0.5):
        picked = list()
        leave_every = int(1 / ratio)
        counter = 1
        for i in freqs:
            if counter % leave_every != 0:
                picked.append(i)
            counter += 1

        # Always add the last one regardless what
        if freqs[-1] not in picked:
            picked.pop()
            picked.append(freqs[-1])

        return picked


class CPUController:
    """ On the Jetson Xavier, the CPU cores are related (check /sys/devices/system/cpu/cpu<id>/cpufreq/related_cpus).
    this means that setting the frequency of one of them sets the frequency for the others as well. Hence, in this
    script we manipulate the state of CPU0 which in turn manipulates the other cores' states.
    Notes:
        - scaling_min_freq and scaling_max_freq set the bounds on the cpu frequency that any governor can choose
        (including userspace governor).

        - scaling_cur_freq is frequency the kernel thinks the CPU is running at.

        - cpuinfo_cur_freq: Current frequency of the CPU as obtained from the hardware, in KHz.
                            This is the frequency the CPU actually runs at.
        - related_cpus: List of Online + Offline CPUs that need software coordination of frequency.

        """

    CPU_CONFIG_DIR = "/sys/devices/system/cpu"

    @staticmethod
    def get_cur_freq():
        return int(Utils.read_first_line_from_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_cur_freq"))

    @staticmethod
    def get_cpu_topology():
        res = dict()
        for i in range(AGX_CPU_CORE_COUNT):
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu{i}/online"
            res[i] = int(Utils.read_first_line_from_file(path))
        return res

    @staticmethod
    def set_cpu_freq_and_bounds(freq):
        order = ["min", "max"]

        # Determine which bound to set first.
        if CPUController.get_cur_freq() < freq:
            order = ["max", "min"]

        # Set the frequency scaling governor to userspace.
        governor = 'userspace'
        CPUController.set_scaling_governor(governor)

        # Set the frequency bounds
        for _t in order:
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_{_t}_freq"
            Utils.write_file(path, str(freq))

        # Set the CPU freq
        CPUController._set_freq(freq)

    @staticmethod
    def set_freq(freq):
        # Set the frequency scaling governor to userspace.
        governor = 'userspace'
        CPUController.set_scaling_governor(governor)
        CPUController._set_freq(freq)

    @staticmethod
    def _set_freq(freq):
        Utils.write_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_setspeed", str(freq))

    @staticmethod
    def set_scaling_governor(governor):
        Utils.write_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_governor", str(governor))

    @staticmethod
    def get_scaling_governor():
        return Utils.read_first_line_from_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_governor")

    @staticmethod
    def set_enabled_cores(num_enabled_cpus):
        """ Enable or disable CPU cores. PLease visit Nvidia documentation @ https://tinyurl.com/y5dkrv7z. """
        for i in range(AGX_CPU_CORE_COUNT):
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu{i}/online"
            if i in range(num_enabled_cpus):
                # Enable this core (change its state to online)
                Utils.write_file(path, "1")
            else:
                # Disable this core.
                Utils.write_file(path, "0")

    @staticmethod
    def get_enabled_cores():
        """ Get number of online CPU cores."""
        count = 0
        for i in range(AGX_CPU_CORE_COUNT):
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu{i}/online"
            if Utils.read_file_as_int(path) == 1:
                count += 1
        return count

    @staticmethod
    def disable_cpu_idle_states():
        for i in range(AGX_CPU_CORE_COUNT):
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu{i}/online"
            status = Utils.read_file_as_int(path)
            if status == 1:
                # Disable per-core idle state.
                CPUController.set_cpu_idle_states(i, 0, 1)

                # Disable per-cluster idle state.
                CPUController.set_cpu_idle_states(i, 1, 1)

    @staticmethod
    def enable_cpu_idle_states():
        for i in range(AGX_CPU_CORE_COUNT):
            path = f"{CPUController.CPU_CONFIG_DIR}/cpu{i}/online"
            status = Utils.read_file_as_int(path)
            if status == 1:
                # Enable the idle state power-gating for the core and its cluster.
                CPUController.set_cpu_idle_states(i, 0, 0)
                CPUController.set_cpu_idle_states(i, 1, 0)

    @staticmethod
    def set_cpu_idle(enable: bool):
        if enable:
            CPUController.enable_cpu_idle_states()
        else:
            CPUController.disable_cpu_idle_states()

    @staticmethod
    def disable_railgate():
        CPUController.set_cpu_idle(False)

    @staticmethod
    def set_cpu_topology(num_enabled_cpus, freq):
        # Enable the cores.
        CPUController.set_enabled_cores(num_enabled_cpus)

        # Then set their frequencies.
        CPUController.set_cpu_freq_and_bounds(freq)

    @staticmethod
    def set_cpu_idle_states(cpu, state, value):
        path = f"{CPUController.CPU_CONFIG_DIR}/cpu{cpu}/cpuidle/state{state}/disable"
        Utils.write_file(path, str(value))

    @staticmethod
    def get_avail_freq():
        if platform.processor() != 'aarch64':
            return Defaults.CPU_FREQUENCIES

        return Utils.read_int_array_and_sort(
            f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_available_frequencies")

    @staticmethod
    def get_max_freq():
        if platform.processor() != 'aarch64':
            return Defaults.CPU_FREQUENCIES[-1]
        return int(Utils.read_first_line_from_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/cpuinfo_max_freq"))

    @staticmethod
    def get_min_freq():
        return int(Utils.read_first_line_from_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/cpuinfo_min_freq"))

    @staticmethod
    def set_freq_upper_bound(freq):

        # We need to make sure that the new upper bound is greater than the lower bound
        lower_bound = CPUController.get_freq_lower_bound()
        if lower_bound > freq:
            # Set lower bound to lowest frequency possible
            least_freq = CPUController.get_min_freq()
            CPUController.set_freq_lower_bound(least_freq)

        # Set the upper bound on the frequency scaling.
        Utils.write_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_max_freq", str(freq))

    @staticmethod
    def set_freq_lower_bound(freq):
        # We need to make sure that the new lower bound is lower than the upper bound
        upper_bound = CPUController.get_freq_upper_bound()
        if upper_bound < freq:
            # Set the upper bound to the highest frequency possible
            highest_freq = CPUController.get_max_freq()
            CPUController.set_freq_upper_bound(highest_freq)

        # Set the lower bound on the frequency scaling
        Utils.write_file(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_min_freq", str(freq))

    @staticmethod
    def get_freq_lower_bound():
        return Utils.read_file_as_int(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_min_freq")

    @staticmethod
    def get_freq_upper_bound():
        return Utils.read_file_as_int(f"{CPUController.CPU_CONFIG_DIR}/cpu0/cpufreq/scaling_max_freq")

    @staticmethod
    def get_num_freqs():
        return len(CPUController.get_avail_freq())

    @staticmethod
    def get_ratio_of_frequencies(ratio=0.5):
        # TODO: check whether 4 here is ok or not.
        freqs = CPUController.get_avail_freq()[4:]
        if int(ratio) == 1:
            return freqs
        picked = Utils.get_ratio_of_frequencies(freqs, ratio)
        return picked

    @staticmethod
    def set_freq_to_max():
        max_freq = CPUController.get_max_freq()
        CPUController.set_freq(max_freq)


class GPUController:
    GPU_CONFIG_DIR = "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b"

    @staticmethod
    def get_avail_freq():
        if platform.processor() != 'aarch64':
            return Defaults.GPU_FREQUENCIES
        return Utils.read_int_array_and_sort(f"{GPUController.GPU_CONFIG_DIR}/available_frequencies")

    @staticmethod
    def get_cur_freq():
        return int(Utils.read_first_line_from_file(f"{GPUController.GPU_CONFIG_DIR}/cur_freq"))

    @staticmethod
    def set_freq_and_bounds(freq):
        order = ["min", "max"]

        if GPUController.get_cur_freq() < freq:
            order = ["max", "min"]

        # Set the frequency scaling governor to userspace.
        governor = 'userspace'
        GPUController.set_scaling_governor(governor)

        for _t in order:
            path = f"{GPUController.GPU_CONFIG_DIR}/{_t}_freq"
            Utils.write_file(path, str(freq))

        # Set the GPU freq
        GPUController._set_freq(freq)

    @staticmethod
    def set_freq(freq):
        # Set the frequency scaling governor to userspace.
        governor = 'userspace'
        GPUController.set_scaling_governor(governor)
        GPUController._set_freq(freq)

    @staticmethod
    def _set_freq(freq):
        Utils.write_file(f"{GPUController.GPU_CONFIG_DIR}/userspace/set_freq", str(freq))

    @staticmethod
    def set_railgate(enable):
        """ Apparently this shuts down the GPU when idle. """
        if type(enable) is int:
            Utils.write_file("/sys/devices/17000000.gv11b/railgate_enable", str(enable))
        elif type(enable) is bool:
            if enable:
                Utils.write_file("/sys/devices/17000000.gv11b/railgate_enable", str(1))
            else:
                Utils.write_file("/sys/devices/17000000.gv11b/railgate_enable", str(0))

    @staticmethod
    def disable_railgate():
        GPUController.set_railgate(False)

    @staticmethod
    def enable_railgate():
        GPUController.set_railgate(True)

    @staticmethod
    def get_railgate():
        return int(Utils.read_first_line_from_file("/sys/devices/17000000.gv11b/railgate_enable"))

    @staticmethod
    def get_scaling_governors():
        return Utils.read_int_array_and_sort(f"{GPUController.GPU_CONFIG_DIR}/available_governors")

    @staticmethod
    def set_scaling_governor(governor):
        Utils.write_file(f"{GPUController.GPU_CONFIG_DIR}/governor", str(governor))

    @staticmethod
    def get_gpu_scaling_governor():
        return Utils.read_first_line_from_file(f"{GPUController.GPU_CONFIG_DIR}/governor")

    @staticmethod
    def get_max_freq():
        return int(Utils.read_int_array_and_sort(f"{GPUController.GPU_CONFIG_DIR}/available_frequencies")[-1])

    @staticmethod
    def get_min_freq():
        return int(Utils.read_int_array_and_sort(f"{GPUController.GPU_CONFIG_DIR}/available_frequencies")[0])

    @staticmethod
    def set_freq_upper_bound(freq):

        # We need to make sure that the new upper bound is greater than the lower bound
        lower_bound = GPUController.get_freq_lower_bound()
        if lower_bound > freq:
            # Set lower bound to lowest frequency possible
            least_freq = GPUController.get_min_freq()
            GPUController.set_freq_lower_bound(least_freq)

        # Set the upper bound on the frequency scaling.
        Utils.write_file(f"{GPUController.GPU_CONFIG_DIR}/max_freq", str(freq))

    @staticmethod
    def set_freq_lower_bound(freq):
        # We need to make sure that the new lower bound is lower than the upper bound
        upper_bound = GPUController.get_freq_upper_bound()
        if upper_bound < freq:
            # Set the upper bound to the highest frequency possible
            highest_freq = GPUController.get_max_freq()
            GPUController.set_freq_upper_bound(highest_freq)

        # Set the lower bound on the frequency scaling
        Utils.write_file(f"{GPUController.GPU_CONFIG_DIR}/min_freq", str(freq))

    @staticmethod
    def get_freq_lower_bound():
        return int(Utils.read_first_line_from_file(f"{GPUController.GPU_CONFIG_DIR}/min_freq"))

    @staticmethod
    def get_freq_upper_bound():
        return int(Utils.read_first_line_from_file(f"{GPUController.GPU_CONFIG_DIR}/max_freq"))

    @staticmethod
    def get_num_freqs():
        return len(GPUController.get_avail_freq())

    @staticmethod
    def get_ratio_of_frequencies(ratio=0.5):
        freqs = GPUController.get_avail_freq()
        if int(ratio) == 1:
            return freqs
        picked = Utils.get_ratio_of_frequencies(freqs, ratio)
        return picked

    @staticmethod
    def set_freq_to_max():
        max_freq = GPUController.get_max_freq()
        GPUController.set_freq(max_freq)


class MemoryController:
    @staticmethod
    def get_avail_freq():
        if platform.processor() != 'aarch64':
            return Defaults.MEMORY_FREQUENCIES

        freqs = [int(i) * 1000 for i in Utils.read_dir("/sys/kernel/debug/bpmp/debug/emc/tables/regular/")]
        freqs.sort()
        return freqs

    @staticmethod
    def get_cur_freq():
        return int(Utils.read_first_line_from_file("/sys/kernel/debug/bpmp/debug/clk/emc/rate"))

    @staticmethod
    def set_freq(freq):
        Utils.write_file("/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked", "1")
        Utils.write_file("/sys/kernel/debug/bpmp/debug/clk/emc/state", "1")
        Utils.write_file("/sys/kernel/debug/bpmp/debug/clk/emc/rate", str(freq))

    @staticmethod
    def get_max_freq():
        if platform.processor() != 'aarch64':
            return Defaults.MEMORY_FREQUENCIES[-1]
        return Utils.read_file_as_int("/sys/kernel/debug/bpmp/debug/clk/emc/max_rate")

    @staticmethod
    def get_min_freq():
        return Utils.read_file_as_int("/sys/kernel/debug/bpmp/debug/clk/emc/min_rate")

    @staticmethod
    def set_freq_to_max():
        max_freq = MemoryController.get_max_freq()
        MemoryController.set_freq(max_freq)

    @staticmethod
    def get_num_freqs():
        return len(MemoryController.get_avail_freq())

    @staticmethod
    def get_ratio_of_frequencies(ratio=0.5):
        freqs = MemoryController.get_avail_freq()
        if int(ratio) == 1:
            return freqs
        picked = Utils.get_ratio_of_frequencies(freqs, ratio)
        return picked


class PowerMonitor:
    # External power monitor
    @staticmethod
    def get_wattsup_reading():
        return Utils.read_subprocess_output(["./wattsup", "-c 1", "ttyUSB0", "watts"])

    @staticmethod
    def get_wattsup_reading_watts():
        reading = PowerMonitor.get_wattsup_reading()
        return float(reading.split(",")[1])

    @staticmethod
    def get_pmu_reading_all_watts():
        res = {}

        gpu = PowerMonitor.get_pmu_reading_gpu_watts()
        cpu = PowerMonitor.get_pmu_reading_cpu_watts()
        soc = PowerMonitor.get_pmu_reading_soc_watts()
        io_sys = PowerMonitor.get_pmu_reading_system_watts()
        ddr = PowerMonitor.get_pmu_reading_ddr_watts()
        cv = PowerMonitor.get_pmu_reading_cv_watts()

        res['gpu'] = gpu
        res['cpu'] = cpu
        res['soc'] = soc

        res['sys'] = io_sys
        res['ddr'] = ddr
        res['cv'] = cv
        res['all'] = gpu + cpu + soc + ddr + io_sys

        return res

    @staticmethod
    def get_pmu_reading_sum_watts():
        gpu = PowerMonitor.get_pmu_reading_gpu_watts()
        cpu = PowerMonitor.get_pmu_reading_cpu_watts()
        soc = PowerMonitor.get_pmu_reading_soc_watts()

        io_sys = PowerMonitor.get_pmu_reading_system_watts()
        ddr = PowerMonitor.get_pmu_reading_ddr_watts()
        cv = PowerMonitor.get_pmu_reading_cv_watts()

        return gpu + cpu + soc + io_sys + ddr + cv

    @staticmethod
    def get_pmu_reading_gpu_watts():

        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/in1_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/curr1_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input") / 1000

    @staticmethod
    def get_pmu_reading_cpu_watts():
        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/in2_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/curr2_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input") / 1000

    @staticmethod
    def get_pmu_reading_soc_watts():
        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/in3_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/curr3_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power2_input") / 1000

    @staticmethod
    def get_pmu_reading_cv_watts():
        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/in1_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/curr1_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device0/in_power0_input") / 1000

    @staticmethod
    def get_pmu_reading_ddr_watts():
        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/in2_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/curr2_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device0/in_power1_input") / 1000

    @staticmethod
    def get_pmu_reading_system_watts():
        if platform.release() == JP5_RELEASE:
            milli_voltage = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/in3_input"
            milli_current = "/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon3/curr3_input"
            return Utils.read_file_as_float(milli_voltage) * Utils.read_file_as_float(milli_current) * 1e-6
        else:
            return Utils.read_file_as_float("/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device0/in_power2_input") / 1000


class ThermalMonitor:
    @staticmethod
    def get_temp_all():
        res = dict()
        res['cpu'] = ThermalMonitor.get_temp_cpu()
        res['gpu'] = ThermalMonitor.get_temp_gpu()
        res['aux'] = ThermalMonitor.get_temp_aux()

        return res

    @staticmethod
    def get_temp_cpu():
        return float(Utils.read_first_line_from_file("/sys/devices/virtual/thermal/thermal_zone0/temp")) / 1000

    @staticmethod
    def get_temp_gpu():
        return float(Utils.read_first_line_from_file("/sys/devices/virtual/thermal/thermal_zone1/temp")) / 1000

    @staticmethod
    def get_temp_aux():
        return float(Utils.read_first_line_from_file("/sys/devices/virtual/thermal/thermal_zone2/temp")) / 1000


class FanController:
    @staticmethod
    def get_config_file():
        return glob.glob("/sys/devices/platform/pwm-fan/hwmon/*/pwm1")[0]

    @staticmethod
    def get_speed():
        if platform.release() == JP5_RELEASE:
            return int(Utils.read_first_line_from_file(FanController.get_config_file()))
        return int(Utils.read_first_line_from_file("/sys/devices/pwm-fan/target_pwm"))

    @staticmethod
    def set_speed(speed):
        if 255 >= int(speed) >= 0:
            # Newer JP 5
            if platform.release() == JP5_RELEASE:
                os.system("systemctl stop nvfancontrol")
                Utils.write_file(FanController.get_config_file(), str(speed))
            else:
                Utils.write_file("/sys/devices/pwm-fan/target_pwm", str(speed))

    @staticmethod
    def set_speed_to_max():
        if str(os.uname().nodename) == 'node-15':
            FanController.set_speed(220)
        else:
            FanController.set_speed(255)


class DVFSHelpers:
    dvfs_levels = ['low', 'mid', 'high']

    @staticmethod
    def generate_random_dvfs_config():
        cpu_freqs = CPUController.get_avail_freq()
        gpu_freqs = GPUController.get_avail_freq()
        memory_freqs = MemoryController.get_avail_freq()
        return {S_CPU_FREQ: pick(cpu_freqs), S_GPU_FREQ: pick(gpu_freqs), S_MEMORY_FREQ: pick(memory_freqs)}

    @staticmethod
    def generate_random_dvfs_config_partial(ratio=0.5):
        cpu_freqs = CPUController.get_ratio_of_frequencies(ratio)
        gpu_freqs = GPUController.get_ratio_of_frequencies(ratio)
        memory_freqs = MemoryController.get_ratio_of_frequencies(ratio)
        return {S_CPU_FREQ: pick(cpu_freqs), S_GPU_FREQ: pick(gpu_freqs), S_MEMORY_FREQ: pick(memory_freqs)}

    @staticmethod
    def get_dvfs_config_id(config):
        return f"{int(config[S_CPU_FREQ])}_{int(config[S_GPU_FREQ])}_{int(config[S_MEMORY_FREQ])}"

    @staticmethod
    def get_dvfs_config_level(configs):
        def internal(_config):
            cpu_freq = int(_config[S_CPU_FREQ])
            gpu_freq = int(_config[S_GPU_FREQ])
            memory_freq = int(_config[S_MEMORY_FREQ])

            if cpu_freq == CPUController.get_avail_freq()[0] and gpu_freq == GPUController.get_avail_freq()[0] and \
                    memory_freq == MemoryController.get_avail_freq()[0]:
                return 'low'
            elif cpu_freq == CPUController.get_avail_freq()[-1] and gpu_freq == GPUController.get_avail_freq()[-1] and \
                    memory_freq == MemoryController.get_avail_freq()[-1]:
                return 'high'
            elif cpu_freq == 1190400 and gpu_freq == 675750000 and memory_freq == 1065600000:
                return 'mid'
            else:
                return f'{cpu_freq}_{gpu_freq}_{memory_freq}'

        if type(configs) is list:
            result = list()
            for config in configs:
                result.append(internal(config))
            return result
        elif type(configs) is dict:
            return internal(configs)
        elif type(configs) is pd.Series:
            return internal(configs.to_dict())

    @staticmethod
    def set_dvfs_config(config):
        if platform.processor() != 'aarch64':
            return
        CPUController.set_freq(config[S_CPU_FREQ])
        GPUController.set_freq(config[S_GPU_FREQ])
        MemoryController.set_freq(config[S_MEMORY_FREQ])

    @staticmethod
    def reset_dvfs_settings():
        BoardController.reset_dvfs_settings()

    @staticmethod
    def verify_dvfs_config(config):
        if platform.processor() != 'aarch64':
            return True

        if CPUController.get_cur_freq() != config[S_CPU_FREQ]:
            logger.error(f"Failed to set CPU frequency to: {config[S_CPU_FREQ]} "
                         f"-- Current is: {CPUController.get_cur_freq()}")
            return False

        if GPUController.get_cur_freq() != config[S_GPU_FREQ]:
            logger.error(f"Failed to set GPU frequency to: {config[S_GPU_FREQ]} "
                         f"-- Current is: {GPUController.get_cur_freq()}")
            return False

        if MemoryController.get_cur_freq() != config[S_MEMORY_FREQ]:
            logger.error(
                f"Failed to set Memory frequency to: {config[S_MEMORY_FREQ]} "
                f"-- Current is: {MemoryController.get_cur_freq()}")
            return False
        return True

    @staticmethod
    def get_a_dvfs_config(dvfs_config_index=-1, dvfs_configs_file_path=None):
        """
         Given a DVFS config index, this method scans a given file and returns the configuration as a tuple of
         the configuration dict, and its string representation. If the idx is -1, or the configuration file path is
         None, we pick a config randomly.

         @returns: dvfs_config_as_dict, dvfs_config_as_str

        """
        if dvfs_config_index == -1 or dvfs_configs_file_path is None:
            logger.info("Picking a random DVFS config")
            # Pick a custom non-seen before DVFS config
            dvfs_config = DVFSHelpers.generate_random_dvfs_config_partial()
            dvfs_str_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
        else:
            logger.info(f"Loading DVFS config {dvfs_config_index} from file")
            # Load the DVFS config from a file
            # Config file contains a configuration to run on each line
            if not os.path.exists(dvfs_configs_file_path):
                dvfs_configs_file_path = os.path.join(DVFS_CONFIGS_DIR, dvfs_configs_file_path)

            with open(dvfs_configs_file_path, 'r') as config_file_obj:
                lines = list()
                for line in config_file_obj.readlines():
                    if line == '\n':
                        continue
                    else:
                        lines.append(json.loads(line))

            if dvfs_config_index >= len(lines):
                logger.error(f"Input index {dvfs_config_index} is >= number of records {len(lines)}")
                return
            dvfs_config = lines[dvfs_config_index]
            dvfs_str_id = DVFSHelpers.get_dvfs_config_id(dvfs_config)
        logger.info(f"-- DVFS config {dvfs_config}")
        return dvfs_config, dvfs_str_id

    @staticmethod
    def is_equal(config_1: dict, config_2: dict):
        if config_1.keys() != config_2.keys():
            return False
        for key in config_1.keys():
            if int(config_1.get(key)) != int(config_2.get(key)):
                return False
        return True

    @staticmethod
    def extract_dvfs_config(input_dict: dict):
        output = dict()
        if S_CPU_FREQ in input_dict.keys():
            if input_dict.get(S_CPU_FREQ) == 'dynamic':
                output[S_CPU_FREQ] = -1
            else:
                output[S_CPU_FREQ] = int(input_dict.get(S_CPU_FREQ))

        if S_GPU_FREQ in input_dict.keys():
            if input_dict.get(S_GPU_FREQ) == 'dynamic':
                output[S_GPU_FREQ] = -1
            else:
                output[S_GPU_FREQ] = int(input_dict.get(S_GPU_FREQ))

        if S_MEMORY_FREQ in input_dict.keys():
            if input_dict.get(S_MEMORY_FREQ) == 'dynamic':
                output[S_MEMORY_FREQ] = -1
            else:
                output[S_MEMORY_FREQ] = int(input_dict.get(S_MEMORY_FREQ))
        return output

    @staticmethod
    def generate_unique_dvfs_configs(configurations_count=None, ratio=1.0):
        """ This method generates a configuration file containing certain number of unique DVFS combinations. """
        # TODO: right now we focus on generating all configs, fix it to be smarter and select representitive ranges
        # when we choose smaller number.

        # TODO: we ignore the first 4 CPU frequencies as they are usually under-provisioned
        cpu_freqs = CPUController.get_ratio_of_frequencies(ratio)
        gpu_freqs = GPUController.get_ratio_of_frequencies(ratio)
        memory_freqs = MemoryController.get_ratio_of_frequencies(ratio)
        n_cpu = len(cpu_freqs)
        n_gpu = len(gpu_freqs)
        n_memory = len(memory_freqs)

        available_count = n_cpu * n_gpu * n_memory
        logger.info(f"Available combinations: {available_count}")
        n = available_count

        if configurations_count is not None:
            n = configurations_count

        if n > available_count:
            logger.error(f"Selected greater than available {configurations_count} > {available_count}")

        content = dict()
        for cpu in cpu_freqs:
            for gpu in gpu_freqs:
                for mem in memory_freqs:
                    key = f'{cpu}_{gpu}_{mem}'
                    val = {S_CPU_FREQ: cpu, S_GPU_FREQ: gpu, S_MEMORY_FREQ: mem}
                    if content.get(key, -1) == -1:
                        content[key] = val

        result = list(content.values())
        return result[0:n]

    @staticmethod
    def generate_unique_dvfs_configs_for_gpu(configurations_count=None, ratio=1.0):
        """ This method generates a configuration file containing certain number of unique DVFS combinations
        fixing CPU and Memory controller to max. """

        cpu_freq = CPUController.get_max_freq()
        memory_freq = MemoryController.get_max_freq()
        gpu_freqs = GPUController.get_ratio_of_frequencies(ratio)
        n_gpu = len(gpu_freqs)
        available_count = n_gpu
        logger.info(f"Available combinations: {available_count}")
        n = available_count

        if configurations_count is not None:
            n = configurations_count

        if n > available_count:
            logger.error(f"Selected greater than available {configurations_count} > {available_count}")

        content = dict()
        for gpu in gpu_freqs:
            key = f'{cpu_freq}_{gpu}_{memory_freq}'
            val = {S_CPU_FREQ: cpu_freq, S_GPU_FREQ: gpu, S_MEMORY_FREQ: memory_freq}
            if content.get(key, -1) == -1:
                content[key] = val
        result = list(content.values())
        return result[0:n]

    @staticmethod
    def load_file(dvfs_file):
        if os.path.exists(dvfs_file):
            p = dvfs_file
        else:
            p = os.path.join(DVFS_CONFIGS_DIR, dvfs_file)
        return Utils.read_file(p)

    @staticmethod
    def count_unique_dvfs_configs(df: pd.DataFrame) -> int:
        return df[[S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ]].apply(lambda x: DVFSHelpers.get_dvfs_config_id(
            x), axis=1).unique().size


if __name__ == "__main__":
    tmp = DVFSHelpers.generate_unique_dvfs_configs(10)
    print(tmp)
