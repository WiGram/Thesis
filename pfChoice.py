"""
Date:    February 2nd, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
The script is a prolongation of EM_NM_EX.py, where we intend
to estimate parameter values based on HMM estimation.
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
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== State simulation ====================== #
# ============================================= #

# --------------------------------------------- #
# Functions
"""
stateDet: Determines state at some point in time

    stateEval: Evaluates which state we are arriving from
        -> say s[t-1] = 2 => stateEval = 2
        -> i = stateEval - 1 due to Python indexing starting at 0
        -> then row 2 (probs[i = 1,:]) belongs to that state
    p:         Uniform random variable (think u when applying)
    probs:     Transition probability matrix
"""

def stateDet(stateEval, p, probs):
    S = len(probs[:,0])
    i = stateEval - 1
    s = [(np.sum(probs[i,:j]) < p <= np.sum(probs[i,:j+1])) * (j + 1) for j in range(S)]
    return np.sum(s)

""" stateSim: Iterates through time and simulations """
def stateSim(s, u, probs, stateSims):
    mat = len(s[0,:])
    for t in range(1,mat):
        s[:,t] = [stateDet(s[m,t-1], u[m,t], probs) for m in range(stateSims)]
    return np.array(s)

def statePaths(stateSims, mat, startstate, probs):
    u = np.random.uniform(0, 1, size = stateSims * mat).reshape(stateSims, mat)
    s = np.repeat(startState, stateSims * mat).reshape(stateSims, mat) # state process

    s = stateSim(s, u, probs, stateSims) # M x T matrix

    return s

# --------------------------------------------- #
# Asset return simulation
"""
Given simulated paths, we can simulate asset returns
"""
def assetReturns(s, states, stateSims, retSims, mu, cov):
    M, N      = stateSims, retSims
    l         = np.array([[np.sum(s[m,:] == i + 1) for i in range(states)] for m in range(M)])
    stateFreq = np.array([sum(l[:,i]) for i in range(states)])

    rets      = [np.random.multivariate_normal(mu[:,i], cov[i], size = stateFreq[i] * M * N) for i in range(states)]

    # For first simulated set of states: simulate 100 return paths
    aR = np.zeros((M * N, assets, mat))
    m = 0
    for i in range(M * N):
        if i > 0 and i % M == 0:
            m += 1
        aR[i] = np.concatenate([rets[j].T[:,i * l[m, j]:l[m,j]*(i + 1)] for j in range(states)], axis = 1)
    
    return aR

# --------------------------------------------- #
# Portfolio returns according to portfolio weighting
"""
Instead of doing a perfect grid search for investment universes of many assets,
common practice is to simulate random portfolio weights and evaluate
"""
def pfWeights(weightSims, assetsPlusBank):
    w   = np.random.random(assetsPlusBank * weightSims).reshape(assetsPlusBank, weightSims)

    """ weights that sum to approximately 1.0 for each column """
    wM = np.array([w[:,i] / np.sum(w[:,i]) for i in range(weightSims)]).T

    return wM

# --------------------------------------------- #
# Weighted returns: the best returns will provide the desired portfolio weights
@jit(nopython = True)
def weightedReturns(stateSims,retSims,rf,aR,wM,weightSims,mat,gamma, aPb):
    M, N, wS = stateSims, retSims, weightSims
    rM       = np.zeros((M * N, wS)) #[r]eturns[M]atrix
    """
    (1) Returns are compounded across time - sum must be along columns: axis = 1
    (2) We are testing all wS different portfolio weights
    """
    for m in range(M * N):
        ret   = [np.sum(wM[:aPb-1,w] * np.exp(np.sum(rf + aR[m], axis = 1))) + wM[aPb-1,w] * np.exp(mat * rf) for w in range(wS)]
        rM[m] = np.array(ret) ** (1 - gamma) / (1 - gamma)
        # Find out why we need to transpose
    
    return np.sum(rM, axis = 0)
# --------------------------------------------- #

# ============================================= #
# ===== Parameter estimation ))))============== #
# ============================================= #

bgn = '2010-01-01'
end = '2015-09-17'

# RF: Risk free rate.
""" (Chosen to be the 52 Wk Coupon equivalent of the 30 day T-bill) """
tbill = quandl.get('USTREASURY/BILLRATES', start_date = bgn, end_date = end)
tbill = tbill['52 Wk Coupon Equiv']
rf = np.array(tbill[0] / 252)

# Asset choice and return computation
sbl = ['AAPL','DJP','HYG','VBMFX','^GSPC'] # Apply alphabetical order.
src = 'yahoo'

close   = web.DataReader(sbl, src, bgn, end)['Close']
prices  = np.array(close)
T       = len(prices[:,0])
exRets  = np.array([np.log(prices[1:,i]) - np.log(prices[:T-1,i]) for i in range(len(sbl))]) - rf
date    = np.array(close.index, dtype = 'datetime64[D]')[1:] # For plotting.

# Model parameter selection
sims    = 50
states  = 3 
assets  = len(sbl)
mat     = len(exRets[0,:])
p       = np.repeat(1.0 / states, states * states)
pS      = np.random.uniform(size = states * mat).reshape(states, mat)

# Running the model and retrieving output
ms, vs, ps, llh, pStar, pStarT = em.EM(exRets, sims, mat, states, assets, p, pS)

"""
Glossary:
ms:     means                             (sims x (assets x states))
vs:     covariance matrices               (sims x (state x (assets x assets)))
ps:     Transition probabilities          (sims x (states x states))
llh:    Log-likelihood                    (sims x 1)
pStar:  Smoothed probabilities            (states x T)
pStarT: Smoothed Transition probabilities (states ** 2 x T)
"""

# Investor choice parameters
""" Final values (sims - 1) are the converged values """
probs = ps[sims-1] # Transition probabilities
cov   = vs[sims-1] # One for each state, size: (States x (Asset x Asset))
mu    = ms[sims-1] # Size (Asset x states)

# ============================================= #
# ===== Implementation ======================== #
# ============================================= #

startState = 1      # 1: Crash, 2: Slow Growth, 3: Bull
mat        = 12     # Investment horizon
gamma      = 5      # Risk aversion
stateSims  = 30     # Simulated state paths
retSims    = 30     # Simulated return processes
weightSims = 10000       # [w]eight[S]ims
aPb        = assets + 1 # [a]ssets[P]lus[B]ank

# --------------------------------------------- #
def findOptimalWeights(startState, mat, gamma, stateSims, retSims, weightSims, aPb):
    s  = statePaths(stateSims, mat, startState, probs)    
    aR = assetReturns(s, states, stateSims, retSims, mu, cov)
    wM = pfWeights(weightSims, aPb)
    wR = weightedReturns(stateSims,retSims,rf,aR,wM,weightSims,mat,gamma, aPb)

    return wM, wR

a, b = findOptimalWeights(startState, mat, gamma, stateSims, retSims, weightSims, aPb)

idx = np.argmax(b)
a[:,idx]