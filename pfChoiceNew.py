"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.
"""

import EM_NM_EX as em
import numpy as np
import pandas as pd
import quandl
from matplotlib import pyplot as plt
from numba import jit
from pandas_datareader import data as web
import genData as gd
import optWeightsModule as owm
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Parameter estimation ))))============== #
# ============================================= #

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
A = len(colNames) # Assets
y = np.array(excessMRets.T) # Returns

trials = 200
S      = 3 # States
T      = len(y[0,:]) # Periods
p      = np.repeat(1.0 / S, S * S).reshape(S, S)
pS     = np.random.uniform(size = S * T).reshape(S, T)

#  Multivariate
ms, vs, ps, llh, pStar, pStarT = em.multEM(y, trials, T, S, A, p, pS)

#  Retrieve parameter estimates
probs = ps[trials-1]  # Transition probabilities
cov   = vs[trials-1]  # One for each state, size: (States x (Asset x Asset))
mu    = ms[trials-1]  # Size (Asset x states)
rf    = rf[T-1]     # Risk-free rate in the last period

# ============================================= #
# ===== Implementation ======================== #
# ============================================= #

start = 1      #  Start state, 1: Crash, 2: Slow Growth, 3: Bull
T     = 12     #  Investment horizon
G     = 5      #  Risk aversion (gamma)
M     = 100    #  Simulated state paths
N     = 1      #  Simulated return processes
W     = 10000  #  [w]eight[S]ims
ApB   = A + 1  #  Assets plus bank account

#  Random number generation
u  = np.random.uniform(0,1,size = (M, T))
w  = np.random.random(size = (ApB, W))

# Read documentation; the following q exits documentation
help(owm.findOptimalWeights)
q

# Run final line to retrieve optimal portfolio weights
eU, uMax, uArgMax, wMax, R, states, wM = owm.findOptimalWeights(M,N,W,T,S,A,rf,G,start,mu,cov,probs,u,w)