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

VAR   NAME                                  DIMENSION       OTHER DESCRIPTION
S     states                                NA              amount of states
A     assets                                NA              amounf of assets
T     mat                                   NA              amount of time periods in series

y     returns                               (A x T)         One row per asset

pS    Smoothed probabilities                (S x T)         alias: pStar
pST   Smoothed Transition probabilities     (S x S x T)     alias: pStarT; 3x3 matrix at each time point

var   Covariance matrix                     (S x A x A)     page 14; one AxA matrix for each state
p_ii  Transition probabilities              (S x S)         page 14; independent of assets
mu    State dependent returns               (A x S)         Analogous to vol

f     Probability density function          (S x T)         Eq. (V13) pg. 13 and 14



# ===== Test run, run the following =========== #

1. runEM.py
2. run the following:

mu     = muFct(pS, y, S, A)
covm   = varFct(pS, y, mu, S, A)
f      = fFct(y, mu, covm, S, A, T)
aR, aS = aFct(T, S, f, p)
bR     = bFct(T, S, f, p)
pStar  = pStarFct(T, S, aR, bR)
pStarT = pStarTFct(f, T, S, aR, aS, bR, p)
p      = pFct(pStarT, S)
l      = logLikFct(y, f, p, pStar, pStarT, S)

test   = multEM(y, sims, T, S, A, p, pS)

"""

# ----- Mean returns -------------------------- #
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
# --------------------------------------------- #

# ----- Covariance matrix --------------------- #
@jit(nopython = True)
def varFct(pS, y, mu, S, A):
    covm = np.zeros((S,A,A))
    for s in range(S):
        for i in range(A):
            for j in range(A):
                covm[s, i, j] = np.sum(pS[s,:] * (y[i,:] - mu[i,s]) * (y[j,:] - mu[j,s])) / np.sum(pS[s,:])
    return covm

def varUniFct(pS, y, mu, S):
    var = np.zeros(S)
    for s in range(S):
        var[s] = np.sum(pS[s,:] * (y - mu[s]) * (y - mu[s])) / np.sum(pS[s,:])
    return var
# --------------------------------------------- #

# ----- Probability density function ---------- #
@jit(nopython = True)
def density(d, dR, covm, A):
    return 1 / np.sqrt( (2 * np.pi) ** A * d) * np.exp(-0.5 * dR.dot(np.linalg.inv(covm)).dot(dR))

@jit(nopython = True)
def densityUni(dR, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * dR ** 2 / var)

# Because of np.linalg.det the (nopython = True) is not supported.
# This has consequences for emMult, which consequently also cannot run
# in (nopython = True)
@jit
def fFct(y, mu, covm, S, A, T):
    """
    d:   determinant
    dR:  demeaned returns
    """
    if A > 1:
        c  = np.linalg.cholesky(covm)
        for s in range(S):
            covm[s] = np.dot(c[s],c[s].T)
        
        d  = np.linalg.det(covm) # returns a determinant for each state
        dR = np.zeros((S,A,T))   # dR: [d]emeaned [R]eturns
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
            f[s,:] = densityUni(dR[s,:], covm[s])

    return f
# --------------------------------------------- #


@jit(nopython = True)
def pFct(pST, S):
    """
    Recollect: p[rows, cols]; cols are past state, rows are approaching state

    # Old code, in case of break down in test run:
    """
    p = np.zeros((S, S))
    den = np.sum(np.sum(pST[:,:,1:], axis = 2), axis = 0)
    for s in range(S):
        p[:,s] = np.sum(pST[:,s,1:], axis = 1) / den[s]

    """    
    p = np.zeros((S, S)) 
    den = np.sum(np.sum(pST[:,:,1:], axis = 2), axis = 0) # den: denominator
    for s in range(S):
        p[:S-1,s] = np.sum(pST[:S-1,s,1:], axis = 1) / den[s] # 1. 2. 3. see below
    p[S-1,:] = 1 - np.sum(p[:S-1,:], axis = 0) # 2. 3. 4. 5.


    1. For each column, build until but not including final row. Final row is a residual
    2. Indexing in Python means e.g. state S = 3 has index 2
    3. Indexing in Python means p[:S-1,:] takes 
        (a) all rows up to, but NOT INCLUDING e.g. 3-1 = 2
        (b) all columns
    4. np.sum(., axis = 0) means, for each column, sum values in all the rows
        i.e. np.sum(p[:2, :], axis = 0) means
            (a) for all columns (:)
            (b) sum values in all rows up until, but NOT INCLUDING row 2
    5. Last row is the residual, not to be estimated directly, but only implicitly
    """

    return p    

# A. Forward algorithm
@jit(nopython = True)
def aFct(T, S, f, p):
    """
    See (V.16) pg. 17 and algorithm pg. 18
    For time t = 0 (in real-speak: period 1, hence the '_1' in a_1)
    a_1  = v * f, where v := 1 / states
    aS_1 = a scaled
    aR_1 = a rescaled
    Initialisation follows below.

    a:   (S x T)
    aS:  (1 x T)
    aR = (S x T)

    NOTE: np.repeat not compatible with (nopython = True)
    """
    a = np.zeros((S, T))

    for s in range(S):
        a[s,:] = f[s, 0] / S

    # Initialisation to be overwritten in the loop further below
    aS = np.zeros(T)
    aR = np.zeros((S, T))

    aS[0]   = np.sum(a[:,0])
    aR[:,0] = a[:,0] / aS[0]

    """
    Fill out for t in [1, T-1], real-speak: [2,T]
    (1) p_ij: transition from i to j
    (2) p[j,i] each column is where we come from, each row where we go to
    (3) Hence: p_ij = p[j,i]
    """
    for t in range(1, T):
        for s in range(S):
            a[s,t] = f[s,t] * np.sum(p[s,:] * aR[:, t-1])

        aS[t]    = np.sum(a[:, t])
        aR[:, t] = a[:,t] / aS[t]
        
    return aR, aS

# B. Backward algorithm
@jit(nopython = True)
def bFct(T, S, f, p):
    """
    Algorithm is backwards recursive
    bR = rescaled b
    Omit bS (scaled) as we will never need it for our purposes (or anyone's?)
    """
    b  = np.ones((S, T))
    bR = np.ones((S, T))
    
    bR[:,T-1] = b[:,T-1] / np.sum(b[:, T - 1])

    """
    (1) p_ij: transition from 'i' to 'j', i.e. p[j, i]
            => p[j,i] each column 'i' is where we come from, each row 'j' where we go to

    Loop: 1. From T-2 (real-speak T-1)
          2. To   -1  (real-speak 0)
          3. Do algorithm and let t_new = t_old - 1
          4. Conclusion: range(T-2, -1, -1)
    """
    
    for t in range(T - 2, -1, -1):
        for i in range(S):
            b[i, t] = np.sum(bR[:,t+1] * f[:, t+1] * p[:, i])
        bR[:, t] = b[:, t] / np.sum(b[:,t])

    return bR

# Output (smoothed) p1, p2, ..., pN
@jit(nopython = True)
def pStarFct(T, S, aR, bR):
    pStar = np.zeros((S, T))
    for t in range(T):
        pStar[:,t] = bR[:, t] * aR[:,t] / np.sum(bR[:, t] * aR[:, t])
    return pStar

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
@jit(nopython = True)
def pStarTFct(f, T, S, aR, aS, bR, p):
    """
    Probabilities are in reverse of notes for this problem,
        inconsistent with the transition probabilities matrix
    
    Notes' definitions:
    p*(i,j)         = smoothed transition probability from 'i' to 'j'
    p_ij            = transition probability from 'i' to 'j'

    Python definitions (consistent with Transition Probability Matrix):
    pStarT[i, j, :] = smoothed transition probability from column (j) to row (i) for all t 
    p[i,j]          = transition probability from column (j) to row (i)
    """
    pStarT = np.zeros((S,S,T))
    pStarT[:,:,0] = p / S # First period
    for t in range(1, T):
        den = aS[t] * np.sum(bR[:,t] * aR[:,t])
        # For every state s in range(S) that we're coming from
        for s in range(S):
            pStarT[:,s,t] = bR[:,t] * f[:,t] * p[:,s] * aR[s, t-1] / den
    
    return pStarT

# E. Expected log-likelihood function to maximise
@jit(nopython = True)
def logLikFct(y, f, p, pS, pST, S):
    c = 1.0 # Arbitrarily set c = some constant, e.g. 1.0 (float)
    
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

    return c + sum_pIJ + sum_pJ

# ============================================= #
# ===== EM Algorithm ========================== #
# ============================================= #

@jit
def multEM(y, sims, T, S, A, p, pS):
    # pS input is only used to initialise the algorithm

    # store variances and probabilities
    llh = np.zeros(sims)
    ps  = np.zeros((sims, S, S))
    vs  = np.zeros((sims, S, A, A))
    ms  = np.zeros((sims, A, S))

    mu  = muFct(pS, y, S, A)
    var = varFct(pS, y, mu, S, A)

    f   = fFct(y, mu, var, S, A, T)

    aR, aS = aFct(T, S, f, p)
    bR     = bFct(T, S, f, p)

    pStar    = pStarFct(T, S, aR, bR)
    pStarT   = pStarTFct(f, T, S, aR, aS, bR, p)

    # 3. EM-loop until convergence (we loop sims amount of times)
    for m in range(sims):
        # Reevaluate parameters given pStar
        mu  = muFct(pStar, y, S, A)
        var = varFct(pStar, y, mu, S, A)
        f   = fFct(y, mu, var, S, A, T)
        p   = pFct(pStarT, S)

        # New smoothed probabilities
        aR, aS = aFct(T, S, f, p)
        bR = bFct(T, S, f, p)

        pStar  = pStarFct(T, S, aR, bR)
        pStarT = pStarTFct(f, T, S, aR, aS, bR, p)
        
        # Compute the log-likelihood to maximise
        logLik = logLikFct(y, f, p, pStar, pStarT, S)

        # Save parameters for later plotting (redundant wrt optimisation)
        ms[m]   = mu
        vs[m]   = var
        ps[m]   = p
        llh[m]  = logLik
    
    return ms, vs, ps, llh, pStar, pStarT


@jit
def uniEM(y, sims, T, S, p, pS):
    # pS input is only used to initialise the algorithm

    # store variances and probabilities
    llh = np.zeros(sims)
    ps  = np.zeros((sims, S, S))
    vs  = np.zeros((sims, S))
    ms  = np.zeros((sims, S))

    mu  = muUniFct(pS, y, S)
    var = varUniFct(pS, y, mu, S)

    f   = fFct(y, mu, var, S, 1, T)

    aR, aS = aFct(T, S, f, p)
    bR     = bFct(T, S, f, p)

    pStar    = pStarFct(T, S, aR, bR)
    pStarT   = pStarTFct(f, T, S, aR, aS, bR, p)

    # 3. EM-loop until convergence (we loop sims amount of times)
    for m in range(sims):
        # Reevaluate parameters given pStar
        mu   = muUniFct(pStar, y, S)
        var  = varUniFct(pStar, y, mu, S)
        f    = fFct(y, mu, var, S, 1, T) # A = 1
        p    = pFct(pStarT, S)

        # New smoothed probabilities
        aR, aS = aFct(T, S, f, p)
        bR = bFct(T, S, f, p)

        pStar  = pStarFct(T, S, aR, bR)
        pStarT = pStarTFct(f, T, S, aR, aS, bR, p)
        
        # Compute the log-likelihood to maximise
        logLik = logLikFct(y, f, p, pStar, pStarT, S)

        # Save parameters for later plotting (redundant wrt optimisation)
        ms[m]   = mu
        vs[m]   = var
        ps[m]   = p
        llh[m]  = logLik
    
    return ms, vs, ps, llh, pStar, pStarT

