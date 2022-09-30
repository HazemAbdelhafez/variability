from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


class PlottingHelpers:
    @staticmethod
    def init():
        PlottingHelpers.set_dpi()
        PlottingHelpers.set_fonts(small=12, medium=16, big=18)
        # plt.subplots_adjust(wspace=0.14, hspace=0.15)
        plt.subplots_adjust(wspace=0.06, hspace=0.2)

    @staticmethod
    def set_dpi():
        # print(plt.rcParams.keys())
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['savefig.pad_inches'] = 0.05

    @staticmethod
    def set_yaxis_label(axs, labels):
        if type(axs) is not list:
            axs = [axs]
        if type(labels) is not list:
            labels = [labels]

        for i, l in zip(axs, labels):
            i.set_ylabel(l)

    @staticmethod
    def set_xaxis_label(axs, labels):
        if type(axs) is not list:
            axs = [axs]
        if type(labels) is not list:
            labels = [labels]

        for i, l in zip(axs, labels):
            i.set_xlabel(l)

    @staticmethod
    def set_fonts(small=12, medium=16, big=18):
        plt.rc('font', size=big)  # controls default text sizes
        plt.rc('axes', titlesize=big)  # fontsize of the axes title
        plt.rc('axes', labelsize=big)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=small)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=small)  # fontsize of the tick labels
        plt.rc('legend', fontsize=medium)  # legend fontsize
        plt.rc('figure', titlesize=big)

    @staticmethod
    def set_font(cfg: dict):
        PlottingHelpers.set_fonts(small=cfg['small'], medium=cfg['medium'], big=cfg['big'])

    @staticmethod
    def capitalize_x_axis_labels(axs):
        if type(axs) is not list:
            axs = [axs]
        for i in axs:
            x_labels = [str(item.get_text()).capitalize() for item in i.get_xticklabels()]
            i.set_xticklabels(x_labels)

    @staticmethod
    def set_to_grey_scale(ax, hue_data=None):
        if hue_data is None:
            n = 1
        else:
            n = len(hue_data)

        _colors = ['lightgray', 'white', 'black', 'red', 'blue']
        _colors = _colors[:min(n, len(_colors))]
        colors = cycle(_colors)
        if n != len(_colors):
            raise Exception(f"Set number of colors correctly to equal number of bars. {n} "
                            f"!= {len(_colors)}")
        new_handles = list()
        for _, patch in enumerate(ax.artists):
            color = next(colors)
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            new_handles.append(Patch(facecolor=color, edgecolor='black'))
        return new_handles

    @staticmethod
    def set_lines_to_grey_scale(ax):
        _colors = ['lightgray', 'white', 'black']
        colors = cycle(_colors)
        new_handles = list()

        for _, patch in enumerate(ax.artists):
            color = next(colors)
            patch.set_color(color)

            new_handles.append(Patch(facecolor=color, edgecolor=color))
        return new_handles

    @staticmethod
    def rotate_x_labels(ax, angle=20):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=angle)

    @staticmethod
    def remove_x_axis_labels_and_ticks(axs):
        if type(axs) is not list:
            axs = [axs]

        for i in axs:
            i.set_xticklabels([])
            i.set_xlabel('')

    @staticmethod
    def remove_y_axis_label(axs):
        if type(axs) is not list:
            axs = [axs]
        for i in axs:
            i.set_ylabel('')

    @staticmethod
    def remove_x_axis_label(axs):
        PlottingHelpers.set_xaxis_label(axs, '')

    @staticmethod
    def remove_legend(axs):
        if type(axs) is not list:
            axs = [axs]
        for ax in axs:
            ax.legend([], [], frameon=False)

    @staticmethod
    def move_y_axis_to_right(axs):
        if type(axs) is not list:
            axs = [axs]

        for ax in axs:
            ax.yaxis.tick_right()

    @staticmethod
    def set_x_axis_ticks(axs, values):
        if type(axs) is not list and type(axs) is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.xaxis.set_ticks(np.array(values))

    @staticmethod
    def set_y_axis_ticks(axs, values):
        if type(axs) is not list and type(axs) is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.yaxis.set_ticks(np.array(values))

    @staticmethod
    def set_y_lim(axs, min_val, max_val):
        if type(axs) is not list and type(axs) is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.set_ylim([min_val, max_val])

    @staticmethod
    def set_x_lim(axs, min_val, max_val):
        if type(axs) is not list and type(axs) is not np.ndarray:
            axs = [axs]

        for ax in axs:
            ax.set_xlim([min_val, max_val])

    @staticmethod
    def set_xlabel(axs, labels):
        if type(axs) is not list and type(axs) is not np.ndarray:
            axs = [axs]

        if type(labels) is not list:
            labels = [labels]

        for ax, label in zip(axs, labels):
            ax.set_xlabel(label)
