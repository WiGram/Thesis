# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram

    * if-else statements seem to obstruct the use of (nopython = True)
    * fFct: here (nopython = True) does not work, and thus I use fFct
            both for the univariate and the multivariate cases, resp.
"""

import plotsModule as pltm
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from numba import jit
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Initial functions ===================== #
# ============================================= #

"""
Terminology

y =  returns
pS = pStar    # Smoothed probabilities
pST = pStarT  # Smoothed Transition probabilities
S = states
A = assets
T = mat       # length of time series

"""

# Output is f1, f2, ..., fN; var, mu must be arrays of parameters (i.e. for each state)
@jit(nopython = True)
def muFct(pS, y, S, A):
    mu = np.zeros((A,S))

    for a in range(A):
        for s in range(S):
            mu[a,s] = np.sum(pS[s,:] * y[a,:]) / np.sum(pS[s,:])
    
    return mu

@jit(nopython = True)
def muUniFct(pS, y, S):
    mu = np.zeros(S)
    for s in range(S):
        mu[s] = np.sum(pS[s,:] * y) / np.sum(pS[s,:])
    return mu

@jit(nopython = True)
def varFct(pS, y, mu, S, A):
    covm = np.zeros((S,A,A))
    for s in range(S):
        for i in range(A):
            for j in range(A):
                covm[s, i, j] = np.sum(pS[s,:] * (y[i,:] - mu[i,s]) * (y[j,:] - mu[j,s])) / np.sum(pS[s,:])
    return covm

def varUniFct(pS, y, mu, S):
    covm = np.zeros(S)
    for s in range(S):
        covm[s] = np.sum(pS[s,:] * (y - mu[s]) * (y - mu[s])) / np.sum(pS[s,:])
    return covm

@jit(nopython = True)
def density(d, dR, covm, A):
    return 1 / np.sqrt( (2 * np.pi) ** A * d) * np.exp(-0.5 * dR.dot(np.linalg.inv(covm)).dot(dR))

@jit(nopython = True)
def densityUni(dR, covm):
    return 1 / np.sqrt(2 * np.pi * covm) * np.exp(-0.5 * dR ** 2 / covm)
    
# 
@jit
def fFct(y, mu, covm, S, A, T):
    """
    d:   determinant
    dR:  demeaned returns
    """
    if A > 1:
        d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
        dR = np.zeros((S,A,T))
        for s in range(S):
            for a in range(A):
                dR[s,a,:] = y[a,:] - mu[a,s]

        f  = np.zeros((S, T))
        for s in range(S):
            for t in range(T):
                f[s, t] = density(d[s], dR[s, :, t], covm[s], A)
    else:
        dR = np.zeros((S, T))
        for s in range(S):
            dR[s,:] = y - mu[s]
        
        f = np.zeros((S, T))
        for s in range(S):
            f[s,:] = densityUni(dR[s,:], covm[s])  # first argument not used in univariates

    return f

# Output: p11, p12, ..., p1N, p21, p22, ..., p2N, ..., pN1, pN2, ..., pNN
@jit(nopython = True)
def pFct(pST, S):
    """
    den: denominator
    """
    p = np.zeros((S, S))
    den = np.sum(np.sum(pST[:,:,1:], axis = 2), axis = 0)
    for s in range(S):
        p[:,s] = np.sum(pST[:,s,1:], axis = 1) / den[s]

    return p    
            
# A. Forward algorithm
@jit
def aFct(T, S, f, p):
    """
    For time t = 0
    a  = v * f, where v := 1 / states; repeat to initialise, reshape to size S x T
    aS = a scaled
    aR = a rescaled
    Initialisation follows below.
    """

    """
    a  = np.ones(S * T).reshape(S, T)
    aS = np.ones(T)
    aR = np.ones(S * T).reshape(S, T)

    a[:,0]  = [f[s,0] / S for s in range(S)]
    aS[0]   = np.sum(a[:,0])
    aR[:,0] = a[:,0] / aS[0]
    """
    a = np.zeros((S, T))

    for s in range(S):
        a[s,:] = f[s, 0] / S

    aS = np.repeat(np.sum(a[:,0]), T)
    aR = np.repeat(a[:,0] / aS[0], T).reshape(S, T)

    """
    Fill out for t in [1, T]
    """
    for t in range(1, T):
        """
        (1) p_ij: transition from i to j
        (2) p[j,i] each column is where we come from, each row where we go to
        (3) Hence: p_ij = p[j,i]
        """
        for s in range(S):
            a[s,t] = f[s,t] * np.sum(p[s,:] * aR[:, t-1])

        aS[t]    = np.sum(a[:, t])
        aR[:, t] = a[:,t] / aS[t]
        
    return aR, aS

# B. Backward algorithm
@jit
def bFct(T, S, f, p):
    """
    bR = rescaled b
    Algorithm is backwards recursive
    """
    b  = np.ones(S * T).reshape(S, T)
    bR = np.repeat(b[:,T-1] / np.sum(b[:, T - 1]), T).reshape(S, T)

    """
    Fill out for t in [T-1: 0]
    range(Start, end, step) <- recall ends at index prior to end.

    Also:
        (1) p_ij: transition from i to j
        (2) p[j,i] each column is where we come from, each row where we go to
        (3) Hence: p_ij = p[j,i]
    """
    for t in range(T - 2, -1, -1):
        for i in range(S):
            b[i, t] = np.sum(bR[:,t+1] * f[:, t+1] * p[:, i])
        bR[:, t] = b[:, t] / np.sum(b[:,t])

    return np.array(bR)

# Output (smoothed) p1, p2, ..., pN
@jit(nopython = True)
def pStarFct(T, S, aR, bR):
    pStar = np.zeros((S, T))
    for t in range(T):
        den   = np.sum(bR[:, t] * aR[:, t]) # den = denominator
        pStar[:,t] = bR[:, t] * aR[:,t] / den
    return pStar

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
@jit(nopython = True)
def pStarTFct(f, T, S, aR, aS, bR, p):
    pStarT = np.zeros((S,S,T))
    pStarT[:,:,0] = p / S
    for t in range(1, T):
        den = aS[t] * np.sum(bR[:,t] * aR[:,t])
        for s in range(S):
            pStarT[s, :,t] = bR[:,t] * f[:,t] * p[:,s] * aR[s, t-1] / den
    
    return pStarT

# E. Expected log-likelihood function to maximise
@jit(nopython = True)
def logLikFct(y, f, p, pS, pST, S):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    
    # first sum (V.13), page 13
    sum_pIJ = 0
    for i in range(S):
        for j in range(S):
            sum_pST = np.sum(pST[j, i, 1:])
            sum_pIJ += np.log(p[j, i]) * sum_pST

    # Second sum (V.13), page 13
    sum_pJ = 0
    for j in range(S):
        sum_pJ += np.sum(pS[j, :] * np.log(f[j, :]))

    return k + sum_pIJ + sum_pJ

# ============================================= #
# ===== EM Algorithm ========================== #
# ============================================= #

@jit
def multEM(returns, sims, mat, S, A, p, pS):
    # pS input is only used to initialise the algorithm

    # store variances and probabilities
    llh = np.zeros(sims)
    ps  = np.zeros((sims, S, S))
    vs  = np.zeros((sims, S, A, A))
    ms  = np.zeros((sims, A, S))

    mu  = muFct(pS, returns, S, A)
    var = varFct(pS, returns, mu, S, A)

    f   = fFct(returns, mu, var, S, A, mat)

    aR, aS = aFct(mat, S, f, p)
    bR     = bFct(mat, S, f, p)

    pStar    = pStarFct(mat, S, aR, bR)
    pStarT   = pStarTFct(f, mat, S, aR, aS, bR, p)

    # 3. EM-loop until convergence (we loop sims amount of times)
    for m in range(sims):
        # Reevaluate parameters given pStar
        mu  = muFct(pStar, returns, S, A)
        var = varFct(pStar, returns, mu, S, A)
        f   = fFct(returns, mu, var, S, A, mat)
        p   = pFct(pStarT, S)

        # New smoothed probabilities
        aR, aS = aFct(mat, S, f, p)
        bR = bFct(mat, S, f, p)

        pStar  = pStarFct(mat, S, aR, bR)
        pStarT = pStarTFct(f, mat, S, aR, aS, bR, p)
        
        # Compute the log-likelihood to maximise
        logLik = logLikFct(returns, f, p, pStar, pStarT, S)

        # Save parameters for later plotting (redundant wrt optimisation)
        ms[m]   = mu
        vs[m]   = var
        ps[m]   = p
        llh[m]  = logLik
    
    return ms, vs, ps, llh, pStar, pStarT


@jit
def uniEM(returns, sims, mat, S, p, pS):
    # pS input is only used to initialise the algorithm

    # store variances and probabilities
    llh = np.zeros(sims)
    ps  = np.zeros((sims, S, S))
    vs  = np.zeros((sims, S))
    ms  = np.zeros((sims, S))

    mu  = muUniFct(pS, returns, S)
    var = varUniFct(pS, returns, mu, S)

    f   = fFct(returns, mu, var, S, 1, mat)

    aR, aS = aFct(mat, S, f, p)
    bR     = bFct(mat, S, f, p)

    pStar    = pStarFct(mat, S, aR, bR)
    pStarT   = pStarTFct(f, mat, S, aR, aS, bR, p)

    # 3. EM-loop until convergence (we loop sims amount of times)
    for m in range(sims):
        # Reevaluate parameters given pStar
        mu   = muUniFct(pStar, returns, S)
        var  = varUniFct(pStar, returns, mu, S)
        f    = fFct(returns, mu, var, S, 1, mat) # A = 1
        p    = pFct(pStarT, S)

        # New smoothed probabilities
        aR, aS = aFct(mat, S, f, p)
        bR = bFct(mat, S, f, p)

        pStar  = pStarFct(mat, S, aR, bR)
        pStarT = pStarTFct(f, mat, S, aR, aS, bR, p)
        
        # Compute the log-likelihood to maximise
        logLik = logLikFct(returns, f, p, pStar, pStarT, S)

        # Save parameters for later plotting (redundant wrt optimisation)
        ms[m]   = mu
        vs[m]   = var
        ps[m]   = p
        llh[m]  = logLik
    
    return ms, vs, ps, llh, pStar, pStarT

