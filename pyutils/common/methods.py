from os.path import join as jp

import numpy as np
import pandas as pd

from pyutils.common.strings import S_RUNTIME_MS, S_TIME_PER_RUN_MS, S_RUNTIME, S_POWER


def is_time_label(label):
    if label == S_RUNTIME_MS:
        return True
    return False


def json_convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def is_power(metric: str):
    metric = metric.lower()
    if metric in ['power', 'watt', 'pwr', 'avg_pwr', 'avg_power', 'p_all']:
        return True
    return False


def is_energy(metric: str):
    metric = metric.lower()
    if metric in ['energy', 'calculated_energy', 'measured_energy', 'joules']:
        return True
    return False


def is_runtime(metric: str):
    metric = metric.lower()
    if metric in [S_RUNTIME, 'time', S_RUNTIME_MS, S_TIME_PER_RUN_MS]:
        return True
    return False


def get_metric(label):
    if is_runtime(label):
        return S_RUNTIME
    elif is_power(label):
        return S_POWER
    else:
        raise Exception(f"Unknown label category: {label}")


def extract_metric_key(metric, keys):
    if is_runtime(metric):
        for key in keys:
            if is_runtime(key):
                return key
    elif is_power(metric):
        for key in keys:
            if is_power(key):
                return key
    elif is_energy(metric):
        for key in keys:
            if is_energy(key):
                return key
    else:
        return metric


def is_modeling(category: str):
    return category in ['modeling']


def is_characterization(category: str):
    return category in ['characterization']


def to_latex(df: pd.DataFrame, label=''):
    latex = df.to_latex(index=False, escape=False, label=label, na_rep='-', multirow=True,
                        multicolumn=True)
    latex = latex.replace("\\\\", "\\\\ \\hline")
    return latex


def parse_bool(value):
    if type(value) is str:
        value = False if value.lower == 'false' else True
        return value
    if type(value) is bool:
        return value
    else:
        return value


def dict_to_columns(item: dict):
    tmp = item.copy()
    for key, value in tmp.items():
        if type(value) is dict:
            for k1, v2 in value.items():
                item[f'{k1}'] = v2
            item.pop(key)
    return item


def flatten(ls):
    return [x for xs in ls for x in xs]


def join(*args):
    return jp(*args)


def check_nan(df: pd.DataFrame):
    count_of_nan_values = df.isnull().values.sum()
    assert count_of_nan_values == 0
