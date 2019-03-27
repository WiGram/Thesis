"""
Date:    March 26th, 2019
Authors: Kristian Strand and William Gram
Subject: Compute Sharpe Ratio on portfolio

Description:
Define functions that compute returns and standard
deviations on a portfolio. Furthermore, design a
function that returns SR, with numerical optimisation
as primary purpose.

Functions:
pfReturns(rM, wM, M, N, W, T)
pfCov(cM, wM, M, N, W, T)
pfSR(pfR, pfS)
pfSRopt(w, rM, cM, M, N, W, T)
numOptSR(w, rM, cM, M, N, W, T)

Imports:
numpy
numba
scipy
"""

import numpy as np
from numba import jit
from scipy import optimize as opt
np.set_printoptions(suppress = True)   # Disable scientific notation

@jit(nopython = True)
def pfRetUniWeight(rM,w,M,N,T):
    """ 
    Description:
        Returns annualised pf returns for a single set of weights.
    
    Args:
        rM: (M*N,A) matrix of returns for each of A assets
        w:  (A,1) vector of normalised portfolio weights
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        T:  Scalar indicating time periods

    Returns:
        pfR:     Scalar of annualised portfolio returns
    """
    pfRets = np.zeros(M*N)
    for i in range(M*N):
        pfRets[i] = np.sum(w * rM[i]) * 12.0/T
    
    return pfRets.mean()


@jit(nopython = True)
def pfCovUniWeight(cM,w,M,N,T):
    """ 
    Description:
        Returns annualised pf standard deviation for a single set
        of weights.
    
    Args:
        cM: (M*N,A,A) vector of AxA matrices of covariances, A = #Assets
        w:  (A,1) vector of normalised portfolio weights
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        T:  Scalar indicating time periods

    Returns:
        pfR:     Scalar of annualised portfolio returns
    """
    pfCovs = np.zeros(M*N)
    for i in range(M*N):
        pfCovs[i] = np.sqrt(w.dot(cM[i].dot(w)) * 12.0 / T)
        
    return pfCovs.mean()

@jit(nopython = True)
def pfReturns(rM, wM, M, N, W, T):
    """ 
    Description:
        Returns annualised pf returns and their argmax and max
    
    Args:
        rM: (M*N,A) matrix of returns for each of A assets
        wM: (W,A) matrix of normalised portfolio weights
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        W:  Scalar indicating portfolio weight simulations
        T:  Scalar indicating time periods

    Returns:
        pfR:     (W,1) Vector of annualised portfolio returns
        argmax:  Scalar of index of max portfolio returns
        max:     Scalar of value of max portfolio returns
        wMax:    (A,1) Vector of maximising portfolio weights
    """
    pfR = np.zeros(W)
    temp    = np.zeros(M*N)
    for j in range(W):
        pfR[j] = pfRetUniWeight(rM,wM[:,j],M,N,T)
    
    return pfR, pfR.argmax(), pfR.max()

@jit(nopython = True)
def pfCov(cM, wM, M, N, W, T):
    """
    Description:
        Returns annualised pf standard deviations and their argmax and max
    
    Args:
        cM: (M*N,A,A) vector of AxA matrices of covariances, A = #Assets
        wM: (W,A) matrix of normalised portfolio weights
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        W:  Scalar indicating portfolio weight simulations
        T:  Scalar indicating time periods

    Returns:
        pfS:     (W,1) Vector of annualised portfolio standard deviation
        argmin:  Scalar of index of min portfolio covariances
        min:     Scalar of value of min portfolio covariances
    """
    pfS = np.zeros(W)
    for j in range(W):
        pfS[j] = pfCovUniWeight(cM,wM[:,j],M,N,T)
    
    return pfS, pfS.argmin(), pfS.min()

@jit(nopython = True)
def pfSR(rM, cM, wM, M, N, W, T):
    """
    Description:
        Returns annualised Sharpe Ratios and their argmax and max.
    
    Args:
        rM: (M*N,A) matrix of returns for each of A assets
        cM: (M*N,A,A) vector of AxA matrices of covariances, A = #Assets
        wM: (W,A) matrix of normalised portfolio weights
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        W:  Scalar indicating portfolio weight simulations
        T:  Scalar indicating time periods
    
    Returns:
        wMax:    (A,1) vector of maximising portfolio weights
        pfSR:    (W,1) vector of portfolio Sharpe Ratios
        argmax:  Scalar indicating index of max Sharpe Ratio
        max:     Scalar indicating value of max Sharpe Ratio
        pfR:     (W,1) vector of portfolio returns
        pfS:     (W,1) vector of portfolio standard deviations
    """
    
    pfR, a,b = pfReturns(rM, wM, M, N, W, T)
    pfS, a,b = pfCov(cM, wM, M, N, W, T)
    
    pfSR = pfR / pfS
    wMax = wM[:,pfSR.argmax()]
    
    return wMax, pfSR.max(), pfSR.argmax(), pfSR, pfR, pfS

@jit(nopython = True)
def pfSRopt(w, rM, cM, M, N, T):
    """
    Description:
        Returns negative of annualised Sharpe Ratios, intended 
        for numerical optimisation with respect to weights.
    
    Args:
        w:  (A,1) vector of normalised portfolio weights
        rM: (M*N,A) matrix of returns for each of A assets
        cM: (M*N,A,A) vector of AxA matrices of covariances, A = #Assets
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        T:  Scalar indicating time periods
    
    Returns:
        pfSR:  Scalar of negative annualised portfolio Sharpe Ratio
    """
    
    pfReturns = np.zeros(M*N)
    for i in range(M*N):
        pfReturns[i] = np.sum(w * rM[i]) * 12.0 / T
    pfReturns = pfReturns.mean()
    
    pfCovariances = np.zeros(M*N)
    for i in range(M*N):
        pfCovariances[i] = np.sqrt(w.dot(cM[i].dot(w)) * 12.0 / T)
    pfCovariances = pfCovariances.mean()
    
    pfSR = pfReturns / pfCovariances
    return -pfSR

def numOptSR(w, rM, cM, M, N, T):
    """
    Description:
        Uses scipy.optimize.minimize to find set of portfolio
        weights that maximises portfolio Sharpe Ratio (SR).
    
    Args:
        w:  (A,1) vector of normalised portfolio weights
        rM: (M*N,A) matrix of returns for each of A assets
        cM: (M*N,A,A) vector of AxA matrices of covariances, A = #Assets
        M:  Scalar indicating simulated state paths
        N:  Scalar indicating simulated return paths per state path
        T:  Scalar indicating time periods
    
    Returns:
        pfSR:  Scalar of annualised portfolio SR
        pfW:   (5,1) vector of SR optimising weights
    """
    args = rM, cM, M, N, T
    A = rM.shape[1]
        
    # Contraints
    def check_sum(weights):
        '''
        Returns 0 if sum of weights is 1.0
        '''
        return np.sum(weights) - 1.0
    
    # By convention of minimize function it should be a function that returns zero for conditions
    cons = ({'type':'eq','fun': check_sum})
    
    # 0-1 bounds for each weight
    bounds = tuple(zip(np.zeros(A),np.ones(A)))
    
    optSRstats = opt.minimize(pfSRopt, w, args, bounds=bounds, constraints=cons)
    wOpt = optSRstats.x
    srOpt = -optSRstats.fun
    
    return wOpt, srOpt

