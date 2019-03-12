python
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
        -> columns: where we are coming from
        -> rows:    where we are going to
"""

# --------------------------------------------- #
# State paths simulation
@jit(nopython = True)
def stateDet(startState, stateSims, p, probs):
    S = len(probs[:,0])
    s = np.zeros((stateSims, S))
    for m in range(stateSims):
        # Each simulation has a t_0 state (python indexing requires '-1')
        i = int(startState[m] - 1)
        for j in range(S):
            # Either the below is false, or true and is then multiplied by (j + 1)
            s[m,j] = (np.sum(probs[:j,i]) < p[m] <= np.sum(probs[:j+1,i])) * (j + 1)
    # The row sums return each simulations state; only one digit is non-zero per row
    return np.sum(s, axis = 1)

""" stateSim: Iterates through time and simulations """
@jit(nopython = True)
def stateSim(startState,u,probs,stateSims,mat):
    s = np.ones((stateSims,mat)) * startState
    for t in range(1,mat):
        s[:,t] = stateDet(s[:,t-1],stateSims,u[:,t-1],probs)
    return s

# --------------------------------------------- #
# Asset return simulation
"""
Given simulated state paths, we can simulate asset returns
"""
def assetReturns(s, states, stateSims, retSims, mu, cov):
    S, M, N   = states,stateSims, retSims
    # Count frequency of each of i states for each of m simulations
    l = np.ones((M,S))
    for m in range(M):
        for i in range(S):
            l[m,i] = np.sum(s[m,:] == i + 1)
    # Count frequency of each state across all simulations
    stateFreq = np.ones(S)
    for i in range(S):
        stateFreq[i] = np.sum(l[:,i])
    #
    rets      = [np.random.multivariate_normal(mu[:,i], cov[i], size = stateFreq[i]) for i in range(S)]
    #
    # For first simulated set of states: simulate 100 return paths
    aR = np.zeros((M * N, assets, mat))
    m = 0
    for i in range(M * N):
        if i > 0 and i % M == 0:
            m += 1
        aR[i] = np.concatenate([rets[j].T[:,i * l[m, j]:l[m,j]*(i + 1)] for j in range(states)], axis = 1)
    #
    return aR

# --------------------------------------------- #
# Portfolio returns according to portfolio weighting
"""
Instead of doing a perfect grid search for investment universes of many assets,
common practice is to simulate random portfolio weights and evaluate
"""
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
rf = np.array(tbill[0] / 252) * 100

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
ms, vs, ps, llh, pStar, pStarT = em.multEM(exRets, sims, mat, states, assets, p, pS)

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
mat        = 24     # Investment horizon
gamma      = 5      # Risk aversion
stateSims  = 100    # Simulated state paths
retSims    = 1      # Simulated return processes
weightSims = 10000  # [w]eight[S]ims
aPb        = A + 1  # [a]ssets[P]lus[B]ank

# --------------------------------------------- #
def findOptimalWeights(stateSims,mat,startState,probs,states,retSims,mu,rf,cov,weightSims,aPb,gamma):
    u  = np.random.uniform(0, 1, size = stateSims * mat).reshape(stateSims, mat)
    w  = np.random.random(aPb * weightSims).reshape(aPb, weightSims)
    s  = stateSim(startState,u,probs,stateSims,mat)
    aR = assetReturns(s, states, stateSims, retSims, mu, cov)
    wM = pfWeights(w, aPb)
    uM = expectedUtility(stateSims,retSims,rf,aR,wM,weightSims,mat,gamma,aPb)

    return wM, wR

a, b = findOptimalWeights(startState, mat, gamma, stateSims, retSims, weightSims, aPb)

idx = np.argmax(b)
a[:,idx]

a = np.linspace(0,10,11)
a[np.argmax(a)]