import numpy as np
from math import sqrt, ceil
from typing import Union
from statsmodels.stats.power import TTestIndPower


def pooledStdev(g1: np.array, g2: np.array) -> float:
    "Calculates pooled standard deviation (Cohen's D) for two groups."
    # Getting each group's size.
    n1, n2 = len(g1), len(g2)
    # Getting each group's variance.
    s1, s2 = g1.var(), g2.var()
    # Calculating pooled standard deviation.
    s = sqrt(
        (((n1 - 1) * s1) + ((n2 - 1) * s2))
        /
        (n1 + n2 - 2)
    )
    return s


def effectSize(g1: np.array, g2: np.array) -> float:
    "Calculates effect size for two groups."
    # Calculating pooled standard deviation.
    s = pooledStdev(g1, g2)
    u1, u2 = g1.mean(), g2.mean()
    # Calculating Cohen's D.
    d = (u1 - u2) / s
    return d


def TTestInd_sampleSize(
        g1: np.array,
        g2: np.array,
        power: float,
        alpha: float = 0.05,
        alternative: str = 'two-sided',
        doround: bool = True
        ) -> Union[float, int]:
    """
    Computes the desired N for an Independent Samples T-Test,
    given alpha, power, and data for both samples.
    """
    # Absolute value of effect size is taken as instructed here:
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestIndPower.solve_power.html
    d = abs(effectSize(g1, g2))
    # We get the ratio of group1 and group2 size.
    ratio = len(g1) / len(g2)
    # Solving independent T-Test sample size for given
    # effect size, alpha, power, and group size ratio.
    nobs1 = TTestIndPower().solve_power(
        nobs1=None,
        effect_size=d,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )  # output is nobs1
    nobs2 = nobs1 * ratio
    N = nobs1 + nobs2  # getting number of subjects needed
    if doround:
        # Rounding up to a whole participant.
        return ceil(N)
    else:
        return N