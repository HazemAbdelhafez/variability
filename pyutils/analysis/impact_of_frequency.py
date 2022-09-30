# Plot on x-axis configs sorted ascending order
# on y-axis: runtime, power, or energy of possible for 3 kernels
# line with dots curve
# make the three observations written in the task list
import pandas as pd
from colour import Color
from matplotlib import pyplot as plt

from pyutils.characterization.networks.utils import VISION_NETWORKS
from pyutils.characterization.networks.utils import get_print_benchmark_name
from pyutils.common.data_interface import DataAttributes, DataHandler
from pyutils.common.experiments_utils import parse_dvfs_configs
from pyutils.common.paths import DATA_DIR, FIGURES_DIR
from pyutils.common.plotting import PlottingHelpers
from pyutils.common.strings import S_RUNTIME_MS, S_DVFS_CONFIG, P_DVFS_CONFIG, S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ

DVFS_file = 'dvfs_3150.json'
dvfs_columns = [S_CPU_FREQ, S_GPU_FREQ, S_MEMORY_FREQ]
dvfs_config_id = S_DVFS_CONFIG + '_id'

dvfs_configs = parse_dvfs_configs(DVFS_file)
lowest_permitted_cpu_freq = dvfs_configs[0][S_CPU_FREQ]
dvfs_configs = pd.DataFrame(dvfs_configs)


def create_dvfs_config_map(_dvfs_configs):
    _dvfs_configs.sort_values(by=dvfs_columns, inplace=True)
    _dvfs_configs[dvfs_config_id] = _dvfs_configs.apply(lambda x: f"{x[S_CPU_FREQ]}_{x[S_GPU_FREQ]}_{x[S_MEMORY_FREQ]}",
                                                        axis=1)
    _dvfs_configs[S_DVFS_CONFIG] = range(_dvfs_configs.shape[0])
    dvfs_configs_map = _dvfs_configs.drop(columns=dvfs_columns).set_index(dvfs_config_id).to_dict()[S_DVFS_CONFIG]
    return dvfs_configs_map


def prepare_nw_data(nw):
    # Create a map from dvfs configuration to dvfs config id
    dvfs_config_map = create_dvfs_config_map(dvfs_configs)

    # Parse network data summary
    attr = DataAttributes(node='node-02', benchmark=nw, return_as='df', metric=S_RUNTIME_MS, overwrite_summary=True)
    data = DataHandler.get_benchmark_data(attr)

    # Make sure the lowest CPU frequency is the lowest we have in the DVFS file.
    data.drop(inplace=True, index=data[data[S_CPU_FREQ] < lowest_permitted_cpu_freq].index)
    data.sort_values(by=dvfs_columns, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data[dvfs_config_id] = \
        data.apply(lambda x: f"{int(x[S_CPU_FREQ])}_{int(x[S_GPU_FREQ])}_{int(x[S_MEMORY_FREQ])}", axis=1)
    data.drop(columns=dvfs_columns, inplace=True)
    # Replace frequency configurations with numbers
    data[S_DVFS_CONFIG] = data.apply(lambda x: dvfs_config_map[x[dvfs_config_id]], axis=1)
    data.drop(columns=[dvfs_config_id], inplace=True)
    # Re-order the df columns
    data = data[[S_DVFS_CONFIG, S_RUNTIME_MS]]
    # Final sorting for sanity
    data.sort_values(by=[S_DVFS_CONFIG], inplace=True)
    data.rename(columns={S_RUNTIME_MS: get_print_benchmark_name(nw)}, inplace=True)

    return data


def combine_nws_data(nws):
    combined = prepare_nw_data(nws[0])
    for nw in nws[1:]:
        nw_data = prepare_nw_data(nw)
        combined = pd.merge(nw_data, combined, on=[S_DVFS_CONFIG])
    return combined


def plot_data(data: pd.DataFrame):
    figures_dir = jp(FIGURES_DIR, 'impact-of-frequency')
    os.makedirs(figures_dir, exist_ok=True)
    figure_path = jp(figures_dir, 'impact_of_frequency.png')

    # Plot a sub-set of the configs
    data = data[(data[S_DVFS_CONFIG] % 100 == 0) | (data[S_DVFS_CONFIG] == data[S_DVFS_CONFIG].max()) |
                (data[S_DVFS_CONFIG] == data[S_DVFS_CONFIG].min())].copy()
    data.reset_index(drop=True, inplace=True)

    data.rename(columns={S_DVFS_CONFIG: P_DVFS_CONFIG}, inplace=True)

    PlottingHelpers.set_fonts(small=12, medium=14, big=16)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    data[P_DVFS_CONFIG] = data[P_DVFS_CONFIG] + 1
    white = Color("white")
    black = Color("black")

    colors = list(black.range_to(white, 21))

    hex_codes = list()
    for i in range(len(colors[:-1])):
        if i % 3 == 0:
            hex_codes.append(colors[i].hex)

    colors = hex_codes
    data.plot(x=P_DVFS_CONFIG, style='.-', ax=ax, color=colors)

    PlottingHelpers.set_y_axis_ticks(ax, values=range(0, 101, 10))
    PlottingHelpers.set_x_lim(ax, min_val=1, max_val=3200)
    PlottingHelpers.set_x_axis_ticks(ax, values=[1, 700] + list(range(1400, 3500, 700)) + [3150])
    ax.set_ylabel("Runtime (ms)")

    PlottingHelpers.set_y_lim(ax, 0, 100)
    ax.grid(True)
    # Set plots whitespace sizes
    plt.subplots_adjust(wspace=0.001, hspace=0.01)
    plt.legend(bbox_to_anchor=(0.5, 1.07), loc='center', ncol=3, frameon=False,
               handletextpad=0.3, borderpad=0.2, labelspacing=0.1, handlelength=0.8)
    plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.05, dpi=600)


if __name__ == '__main__':
    from os.path import join as jp
    import os

    saved_data_file = jp(DATA_DIR, 'impact-of-frequency', 'combined_data.csv')
    if not os.path.exists(saved_data_file):
        df = combine_nws_data(
            nws=[i for i in VISION_NETWORKS if i not in ['inception3', 'densenet', 'vgg', 'mobilenetv2']])
        df.to_csv(saved_data_file, index=False)
    else:
        df = pd.read_csv(saved_data_file)
    plot_data(df)
