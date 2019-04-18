from numba import jit
import numpy as np


@jit(nopython=True)
def expectedUtilityMult(w, returns, rf, g, A, T):
    """
    Description
    -----------------------------
    Computes expected utility of wealth.
    Wealth is compounded return on risky and risk free assets.
    Utility is (W)^(1-gamma) / (1-gamma) -> gamma != 1 !!

    Arguments
    -----------------------------
    w           (A+1,1) vector of standardised weights, default is (.4,.25,.35)
    returns     (M*N,A,T) multivariate simulations, in percent, not decimals
    rf          Scalar in percent, not decimals, default is 0.3 (percent)
    g           Scalar indicating gamma, risk aversion, default is 5
    A           Scalar indicating amount of assets excl. bank account
    T           Periods in months, default is 120
    """

    # rfCR: risk free compounded return
    # rCR:  risky compounded return

    rfCR = np.exp(T * rf/100)
    denominator = 1 - g
    rCR = np.exp(np.sum(returns/100, axis=2))*rfCR
    numerator = (w[A] * rfCR + np.sum(w[:A] * rCR, axis=1)) ** (1 - g)
    return -np.mean(numerator / denominator) * 100000
