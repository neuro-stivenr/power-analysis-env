import numpy as np
from math import sqrt, ceil
from typing import Union
from statsmodels.stats.power import TTestIndPower


def pooledStdev(g1: np.array, g2: np.array) -> float:
    "Calculates pooled standard deviation for two groups."
    # Getting each group's size.
    n1, n2 = g1.size, g2.size
    # Getting each group's standard deviation.
    sd1, sd2 = g1.std(), g2.std()
    # Calculating pooled standard deviation.
    s = sqrt(
        (((n1 - 1) * (sd1**2)) + ((n2 - 1) * (sd2**2)))
        /
        (n1 + n2 - 2)
    )
    return s


def effectSize(g1: np.array, g2: np.array) -> float:
    "Calculates effect size for two groups."
    # Calculating pooled standard deviation.
    s = pooledStdev(g1, g2)
    m1, m2 = g1.mean(), g2.mean()
    # Calculating Cohen's D.
    d = (m1 - m2) / s
    return abs(d)


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
    d = effectSize(g1, g2)
    # We get the ratio of group2 to group1
    ratio = g2.size / g1.size
    # Solving independent T-Test sample size for given
    # effect size, alpha, power, and group size ratio.
    nobs1 = TTestIndPower().solve_power(
        nobs1=None,  # marked as None because we want this value
        effect_size=d,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )  # output is number of observations needed in group 1
    nobs2 = nobs1 * ratio  # to get number of observations needed in group 2 we multiply by the ratio
    N = nobs1 + nobs2  # getting number of subjects needed in total
    if doround:
        # Rounding up to a whole participant.
        return ceil(N)  # needed because multiplying by decimal to get nobs2
    else:
        return N