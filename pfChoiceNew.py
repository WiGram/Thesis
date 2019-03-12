python
"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.
"""

import likelihoodModule as llm
import plotsModule as pltm
import EM_NM_EX as em
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numba import jit
from pandas_datareader import data as web
import genData as gd
np.set_printoptions(suppress = True)   # Disable scientific notation

@jit(nopython = True)
def stateSim3(startState, probs, T, u):
    state_ms = np.ones(T) * startState
    for t in range(1,T):
        state_ms[t] = \
            (state_ms[t-1] == 1) * (
                (u[t] <= probs[0,0]) * 1 + \
                (np.sum(probs[:1,0]) < u[t] <= np.sum(probs[:2,0])) * 2 + \
                (np.sum(probs[:2,0]) < u[t] <= 1) * 3
            ) + (state_ms[t-1] == 2) * (
                (u[t] <= probs[0,1]) * 1 + \
                (np.sum(probs[:1,1]) < u[t] <= np.sum(probs[:2,1])) * 2 + \
                (np.sum(probs[:2,1]) < u[t] <= 1) * 3
            ) + (state_ms[t-1] == 3) * (
                (u[t] <= probs[0,2]) * 1 + \
                (np.sum(probs[:1,2]) < u[t] <= np.sum(probs[:2,2])) * 2 + \
                (np.sum(probs[:2,2]) < u[t] <= 1) * 3
            )
    #
    lenOne = np.sum(state_ms == 1)
    lenTwo = np.sum(state_ms == 2)
    lenThr = np.sum(state_ms == 3)
    #
    length = np.array((lenOne, lenTwo, lenThr))
    #
    return state_ms, length


@jit
def returnSim3(stateSims, returnSims, S, A, startState, mu, cov, probs, T, u):
    rets = np.ones((stateSims*returnSims,A,T))
    #
    for m in range(stateSims):
        st, length = stateSim3(startState, probs, T, u[m])
        for n in range(returnSims):
            for s in range(S):
                rets[m * returnSims + n,:, st == s + 1] = np.random.multivariate_normal(mu[:,s], cov[s], length[s])
    #
    return rets


@jit(nopython = True)
def pfWeights(w, assetsPlusBank):
    """ weights that sum to approximately 1.0 for each column """
    weightSims = w.shape[1]
    wM = np.ones((assetsPlusBank, weightSims))
    for i in range(weightSims):
        wM[:,i] = w[:,i] / np.sum(w[:,i])
    #
    return wM

# --------------------------------------------- #
# Weighted returns: the best returns will provide the desired portfolio weights
@jit(nopython = True)
def expectedUtility(stateSims,retSims,rf,aR,wM,weightSims,mat,gamma,aPb):
    M, N, wS = stateSims, retSims, weightSims
    uM       = np.zeros((wS, M * N)) #[u]tility[M]atrix
    """
    (1) Returns are compounded across time - sum must be along columns: axis = 1
    (2) We are testing all wS different portfolio weights
    """
    for w in range(wS):
        print(w)
        for n in range(M * N):
            uM[w,n]   = (wM[aPb-1,w] * np.exp(mat * rf / 100) + np.sum(wM[:aPb-1,w] * np.exp(np.sum((rf + aR[n,:,:mat])/100.0, axis = 1)))) ** (1 - gamma) / (1 - gamma)
    #
    uM = np.sum(uM, axis = 1) / (M * N)
    #
    return uM

# ============================================= #
# ===== Parameter estimation ))))============== #
# ============================================= #

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

"""
# ===== Monthly Absolute ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
monthlyRets = monthlyRets.drop(['S&P 500'], axis = 1)
colNames =.columns
assets = len(colNames)
y = np.array(monthlyRets.T)
"""

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
A = len(colNames) # Assets
y = np.array(excessMRets.T) # Returns

sims = 200
S    = 3 # States
T    = len(y[0,:]) # Periods
p    = np.repeat(1.0 / S, S * S).reshape(S, S)
pS   = np.random.uniform(size = S * T).reshape(S, T)

#  Multivariate
ms, vs, ps, llh, pStar, pStarT = em.multEM(y, sims, T, S, A, p, pS)

#  Retrieve parameter estimates
probs = ps[sims-1]  # Transition probabilities
cov   = vs[sims-1]  # One for each state, size: (States x (Asset x Asset))
mu    = ms[sims-1]  # Size (Asset x states)
rf    = rf[T-1]     # Risk-free rate in the last period
# ============================================= #
# ===== Implementation ======================== #
# ============================================= #

startState = 1      #  1: Crash, 2: Slow Growth, 3: Bull
T          = 12     #  Investment horizon
gamma      = 5      #  Risk aversion
stateSims  = 100    #  Simulated state paths
retSims    = 1      #  Simulated return processes
weightSims = 10000  #  [w]eight[S]ims
aPb        = A + 1  #  [a]ssets[P]lus[B]ank

#  Random number generation
u  = np.random.uniform(0, 1, size = stateSims * T).reshape(stateSims, T)
w  = np.random.random(aPb * weightSims).reshape(aPb, weightSims)


rets    = returnSim3(stateSims, returnSims, S, A, startState, mu, cov, probs, T, u)
weights = pfWeights(w, aPb)  #  Normed weights
expUtil = expectedUtility(stateSims,retSims,rf,rets,weights,weightSims,mat,gamma,aPb)