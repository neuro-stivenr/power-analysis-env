import numpy as np
from math import sqrt, ceil
from typing import Callable, Union
from statsmodels.stats.power import TTestIndPower

def pooledStdev(g1:np.array, g2:np.array) -> float:
    "Calculates pooled standard deviation (Cohen's D) for two groups."
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.var(), g2.var()
    s = sqrt(
        (((n1 - 1) * s1) + ((n2 - 1) * s2))
        /
        (n1 + n2 - 2)
    )
    return s

def effectSize(g1:np.array, g2:np.array) -> float:
    "Calculates effect size for two groups."
    s = pooledStdev(g1, g2)
    u1, u2 = g1.mean(), g2.mean()
    d = (u1 - u2) / s
    return d
    
# Not sure what the `ratio` and `alternative` parameters do.
def TTestInd_sampleSize(
    g1:np.array, 
    g2:np.array,
    alpha:float=0.05,
    power:float=0.8,
    ratio:float=1,
    alternative:str='two-sided',
    doround:bool=True
    ) -> Union[float,int]:
    """
    Computes the desired sample size within each group for given alpha and power. 
    Effect size is estimated (by effectSize function).
    """
    d = effectSize(g1, g2)
    power_analysis = TTestIndPower()
    n = power_analysis.solve_power(
        effect_size=d,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative
    )
    if doround: return ceil(n)
    else: return n
