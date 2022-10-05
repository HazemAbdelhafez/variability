import pandas as pd

from pyutils.analysis.variability.strings import S_MEDIAN_OF_ALL
from pyutils.common.config import S_CPU_FREQ, S_MEMORY_FREQ, S_GPU_FREQ


class NetworkAnalysisMethods:
    @staticmethod
    def calculate_and_set_group_median(df: pd.DataFrame, filter_values, median, group_col=S_MEDIAN_OF_ALL):
        df.loc[(df[S_CPU_FREQ] == filter_values[0]) & (df[S_GPU_FREQ] == filter_values[1]) &
               (df[S_MEMORY_FREQ] == filter_values[2]),
               [group_col]] = median

    @staticmethod
    def create_key_from_record(record):
        return f"{record[S_CPU_FREQ]}_{record[S_GPU_FREQ]}_{record[S_MEMORY_FREQ]}"

    @staticmethod
    def create_dict_from_key(key):
        key_items = key.split('_')
        return {S_CPU_FREQ: key_items[0], S_GPU_FREQ: key_items[1], S_MEMORY_FREQ: key_items[2]}
