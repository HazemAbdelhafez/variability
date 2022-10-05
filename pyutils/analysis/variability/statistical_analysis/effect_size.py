import numpy as np
from numpy import sqrt
from scipy import stats
from scipy.stats.mstats_basic import winsorize


class EffectSize:
    @staticmethod
    def cohen_d(d1: list, d2: list):
        n1 = len(d1)
        n2 = len(d2)
        d_cohen_den = (n1 - 1) * (np.std(d1, ddof=1) ** 2) + (n2 - 1) * (
                np.std(d2, ddof=1) ** 2)
        d_cohen_den = sqrt(d_cohen_den / (n1 + n2 - 2))
        d_cohen_val = (np.mean(d1) - np.mean(d2)) / d_cohen_den
        return d_cohen_val

    @staticmethod
    def scaled_robust_d(d1: list, d2: list):
        trim_ratio = 0.2
        wins_d1 = winsorize(d1, limits=(trim_ratio, trim_ratio), inplace=False).astype(np.longdouble)
        wins_d2 = winsorize(d2, limits=(trim_ratio, trim_ratio), inplace=False).astype(np.longdouble)
        n1 = len(d1)
        n2 = len(d2)
        y1 = stats.trim_mean(d1, proportiontocut=trim_ratio)
        y2 = stats.trim_mean(d2, proportiontocut=trim_ratio)
        st1 = np.var(wins_d1, ddof=1)
        st2 = np.var(wins_d2, ddof=1)
        numerator = (n1 - 1) * st1 + (n2 - 1) * st2
        stp = sqrt(numerator / (n1 + n2 - 2))
        if y1 == y2:
            return 0

        if stp == 0:
            # Sometimes winsorize will replace all values because of how the thresholds are set
            # this leads to a divide-by-zero error because the std is zero now. I set it manually to -inf
            # to avoid
            # warning messages.
            dr_star = float('-inf')
        else:
            dr_star = (y1 - y2) / stp
        dr = 0.642 * dr_star
        return np.float(dr)


COHEN_D = 'cohen_d'
ROBUST_D = 'robust_d'
