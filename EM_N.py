# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Initial functions ===================== #
# ============================================= #

# Output is f1, f2, ..., fN; vol must be an array of all volatilities
def fFct(returns, vol):
    f = [1 / np.sqrt(2 * np.pi * v) * np.exp(-0.5 * returns ** 2 / v) for v in vol]
    return np.array(f)

# Output: v1^2, v2^2, ..., vN^2
def varFct(pStar, returns, states):
    s = [sum(pStar[s, :] * returns ** 2) / sum(pStar[s, :]) for s in range(states)]
    return np.array(s)

# Output: p11, p12, ..., p1N, p21, p22, ..., p2N, ..., pN1, pN2, ..., pNN
def pFct(pStarT, states):
    n   = states
    den = [sum([sum(pStarT[s * n + i,:]) for i in range(states)]) for s in range(states)]
    p   = [sum(pStarT[s * n + i,:]) / den[s] for s in range(states) for i in range(states)]
    return np.array(p)

# A. Forward algorithm
def aFct(mat, states, f, p):
    a   = [f[i][0] / states for i in range(states)]  # v_j = 1/N, N = states.
    a   = np.repeat(a, mat).reshape(states, mat)
    a_s = np.ones(mat)                               # a_scale
    a_r = np.ones(states * mat).reshape(states, mat) # a_rescale

    # t = 0
    a_s[0]    = sum(a[:,0])
    a_r[:, 0] = a[:,0] / a_s[0]

    # t in [1, T]
    for t in range(1, mat):
        a[:, t]   = [f[j][t] * sum([p[i * states + j] * a_r[i, t-1] for i in range(states)]) for j in range(states)]
        a_s[t]    = sum(a[:, t])
        a_r[:, t] = a[:,t] / a_s[t]

    return np.array(a_r), np.array(a_s)

# B. Backward algorithm
def bFct(mat, states, f, p):    
    b   = np.ones(states * mat).reshape(states, mat)
    b_s = np.ones(mat)                               # b_scale
    b_r = np.ones(states * mat).reshape(states, mat) # b_rescale

    # t = T (= mat - 1)
    b_s[mat-1]      = sum(b[:, mat - 1])
    b_r[:, mat - 1] = b[:, mat - 1] / b_s[mat - 1]

    # t in [0, T - 1] (= mat - 2, stops at previous index, i.e. 0)
    for t in range(mat - 2, -1, -1):
        b[:, t]   = [sum([b_r[s, t+1] * f[s][t+1] * p[i * states + s] for s in range(states)]) for i in range(states)]
        b_s[t]    = sum(b[:,t])
        b_r[:, t] = b[:, t] / b_s[t]

    return np.array(b_r)

# Output (smoothed) p1, p2, ..., pN
def pStarFct(mat, states, a_r, b_r):
    den   = sum([b_r[s, :] * a_r[s, :] for s in range(states)])
    pStar = [b_r[s, :] * a_r[s,:] / den for s in range(states)]
    return np.array(pStar)

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
def pStarTFct(mat, states, a_r, a_s, b_r, p):
    pStarT = np.ones(states * states * mat).reshape(states * states, mat)

    den   = sum([b_r[s, :] * a_r[s, :] for s in range(states)]) * a_s
    pStarT[:, 0] = p / states
    pStarT[:, 1:] = [b_r[s, 1:] * f[s][1:] * p[i * states + s] * a_r[i, :mat - 1] / den[1:] for i in range(states) for s in range(states)]
    return np.array(pStarT)

# E. Expected log-likelihood function to maximise
def logLikFct(vol, p, pStar, pStarT):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    a = sum([sum([np.log(p[s * states + i]) * sum(pStarT[s * states + i, 1:]) for i in range(states)]) for s in range(states)])
    b = sum([-0.5 * sum(pStar[s, :] * (np.log(vol[s]) + y ** 2 / vol[s])) for s in range(states)])
    return k + a + b

# ============================================= #
# ===== Start running the programme =========== #
# ============================================= #

# 0. Load S&P 500 data
# sp500 = pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx')
# d     = np.array(sp500['Date'][15096:], dtype = 'datetime64[D]')
# y     = np.array(sp500['log-ret_x100'][15096:]) # returns

sbl = '^GSPC'
bgn = '2010-01-01'
end = '2015-09-17'
src = 'yahoo'

sp500 = web.DataReader(sbl, src, bgn, end)['Close'].to_frame()
# sp500 = web.DataReader(sbl, src, bgn, end)['Close'].to_frame().resample('MS').mean().round() # Monthly
close = np.array(sp500['Close'])
d     = np.array(sp500.index, dtype = 'datetime64[D]')[1:]
mat   = len(close)
y     = (np.log(close[1:]) - np.log(close[:mat-1])) * 100

# 1. Set initial parameters

mat      = len(y)
states   = 2
sims     = 500
llh      = np.zeros(sims)

# store variances and probabilities
vs       = np.zeros(states * sims).reshape(states, sims)
ps       = np.zeros(states * states * sims). reshape(states * states, sims)

# var won't work with e.g. np.ones(states), hence the "weird" construction
var = 1. + np.random.uniform(size = states)
p   = np.repeat(1.0 / states, states * states)

f       = fFct(y, var)

a_r, a_s = aFct(mat, states, f, p)
b_r      = bFct(mat, states, f, p)

pStar    = pStarFct(mat, states, a_r, b_r)
pStarT   = pStarTFct(mat, states, a_r, a_s, b_r, p)
# Initial checks that all is OK
'OK: Minimum is weakly larger than 0' if min([min(pStar[i,:]) >= 0.0 for i in range(states)])                     else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStar[i,:]) <= 1.0 for i in range(states)])                     else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(sum([pStar[i,:] for i in range(states)]), 10) == 1.0)       else "Error: Probabilities don't sum to 1"
'OK: Minimum is weakly larger than 0' if min([min(pStarT[i,:]) >= 0.0 for i in range(states ** 2)])               else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStarT[i,:]) <= 1.0 for i in range(states ** 2)])               else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(sum([pStarT[i,:] for i in range(states ** 2)]), 10) == 1.0) else "Error: Probabilities don't sum to 1"


# 3. EM-loop until convergence (we loop sims amount of times)
for m in range(sims):
    # Reevaluate parameters given pStar    
    var  = varFct(pStar, y, states)
    f    = fFct(y, var)
    p    = pFct(pStarT, states)

    # New smoothed probabilities
    a_r, a_s = aFct(mat, states, f, p)
    b_r = bFct(mat, states, f, p)

    pStar  = pStarFct(mat, states, a_r, b_r)
    pStarT = pStarTFct(mat, states, a_r, a_s, b_r, p)
    
    # Compute the log-likelihood to maximise
    logLik = logLikFct(var, p, pStar, pStarT)

    # Save parameters for later plotting (redundant wrt optimisation)
    vs[:, m] = var
    ps[:, m] = p
    llh[m] = logLik

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

if states == 2:
    pltm.plotDuo(range(sims), vs[0,:], vs[1,:], 'Var_1', 'Var_2', 'Trials', 'Variance')
    pltm.plotDuo(range(sims), ps[0,:], ps[3,:], 'p11', 'p22', 'Trials', 'Probability')
elif states == 3:
    pltm.plotTri(range(sims), vs[0,:], vs[1,:], vs[2,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance')
    pltm.plotTri(range(sims), ps[0,:], ps[4,:], ps[8,:], 'Trials', 'p11', 'p22', 'p33', 'Probability')
    pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
elif states == 4:
    pltm.plotQuad(range(sims), vs[0,:], vs[1,:], vs[2,:], vs[3,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Var_4', 'Variance')
    pltm.plotQuad(range(sims), ps[0,:], ps[5,:], ps[10,:], ps[15,:], 'Trials', 'p11', 'p22', 'p33', 'p44', 'Probability')

pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

# Generalise plots