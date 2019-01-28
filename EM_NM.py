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
from numba import jit
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Initial functions ===================== #
# ============================================= #

# Output is f1, f2, ..., fN; var, mu must be arrays of parameters (i.e. for each state)
#@jit
def muFct(pStar, returns, states):
    mu  = [np.array([sum(pStar[s, :] * returns[i,:]) / sum(pStar[s, :]) for s in range(states)]) for i in range(len(returns[:,0]))]
    return np.array(mu)

# Output: v1^2, v2^2, ..., vN^2
#@jit
def varFct(pStar, returns, mu, states):
    covm = [np.array([sum(pStar[s, :] * (returns[i,:] - mu[i,s]) * (returns[j,:] - mu[j,s])) / sum(pStar[s, :]) for i in range(assets) for j in range(assets)]).reshape(assets,assets) for s in range(states)]
    return np.array(covm)

#@jit
def fFct(returns, mu, covm, states, assets):
    det             = np.linalg.det(covm) # returns [det1, det2, ..., detN]
    demeanedReturns = np.array([ np.array([returns[i,:] - mu[i, s] for i in range(assets)]) for s in range(states)])
    f = [np.array([1 / np.sqrt( (2 * np.pi) ** assets * det[s]) * np.exp(-0.5 * demeanedReturns[s,:,t].dot(np.linalg.inv(covm[s])).dot(demeanedReturns[s,:,t])) for s in range(states)]) for t in range(mat)]
    return np.array(f)

# Output: p11, p12, ..., p1N, p21, p22, ..., p2N, ..., pN1, pN2, ..., pNN
#@jit
def pFct(pStarT, states):
    n   = states
    den = [sum([sum(pStarT[s * n + i,:]) for i in range(states)]) for s in range(states)]
    p   = [sum(pStarT[s * n + i,:]) / den[s] for s in range(states) for i in range(states)]
    return np.array(p)

# A. Forward algorithm
#@jit
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
        a[:, t]   = [f[t,s] * sum([p[S * states + s] * a_r[S, t-1] for S in range(states)]) for s in range(states)]
        a_s[t]    = sum(a[:, t])
        a_r[:, t] = a[:,t] / a_s[t]

    return np.array(a_r), np.array(a_s)

# B. Backward algorithm
#@jit
def bFct(mat, states, f, p):    
    b   = np.ones(states * mat).reshape(states, mat)
    b_s = np.ones(mat)                               # b_scale
    b_r = np.ones(states * mat).reshape(states, mat) # b_rescale

    # t = T (= mat - 1)
    b_s[mat-1]      = sum(b[:, mat - 1])
    b_r[:, mat - 1] = b[:, mat - 1] / b_s[mat - 1]

    # t in [0, T - 1] (= mat - 2, stops at previous index, i.e. 0)
    for t in range(mat - 2, -1, -1):
        b[:, t]   = [sum([b_r[S, t+1] * f[t+1,S] * p[s * states + S] for S in range(states)]) for s in range(states)]
        b_s[t]    = sum(b[:,t])
        b_r[:, t] = b[:, t] / b_s[t]

    return np.array(b_r)

# Output (smoothed) p1, p2, ..., pN
#@jit
def pStarFct(mat, states, a_r, b_r):
    den   = sum([b_r[s, :] * a_r[s, :] for s in range(states)])
    pStar = [b_r[s, :] * a_r[s,:] / den for s in range(states)]
    return np.array(pStar)

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
#@jit
def pStarTFct(f, mat, states, a_r, a_s, b_r, p):
    pStarT = np.ones(states * states * mat).reshape(states * states, mat)

    den   = sum([b_r[s, :] * a_r[s, :] for s in range(states)]) * a_s
    pStarT[:, 0] = p / states
    pStarT[:, 1:] = [b_r[s, 1:] * f[1:,s] * p[S * states + s] * a_r[S, :mat - 1] / den[1:] for S in range(states) for s in range(states)]
    return np.array(pStarT)

# E. Expected log-likelihood function to maximise
#@jit
def logLikFct(returns, mu, covm, p, pStar, pStarT, f):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    a = sum([sum([np.log(p[s * states + S]) * sum(pStarT[s * states + S, 1:]) for S in range(states)]) for s in range(states)])
    b = sum([-0.5 * sum(pStar[s, :] * f[:, s]) for s in range(states)])
    return k + a + b


# ============================================= #
# ===== Start running the programme =========== #
# ============================================= #

# 0. Load S&P 500 data
# sp500 = pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx')
# d     = np.array(sp500['Date'][15096:], dtype = 'datetime64[D]')
# y     = np.array(sp500['log-ret_x100'][15096:]) # returns

sbl = ['AAPL','DJP','HYG','VBMFX','^GSPC'] # S&P to be last (possibly due to '^')
bgn = '2010-01-01'
end = '2015-09-17'
src = 'yahoo'

# Returns apple first, S&P second..
close = web.DataReader(sbl, src, bgn, end)['Close']
# sp500 = web.DataReader(sbl, src, bgn, end)['Close'].to_frame().resample('MS').mean().round() # Monthly
prices  = np.array(close)
d       = np.array(close.index, dtype = 'datetime64[D]')[1:]
mat     = len(prices[:,0])
returns = np.array([(np.log(prices[1:,i]) - np.log(prices[:mat-1,i])) * 100 for i in range(len(sbl))])

sims     = 100
states   = 3
assets   = len(sbl)

# ============================================= #
# ===== EM Algorithm ========================== #
# ============================================= #

# 1. Set initial parameters
# def runEstimation(returns, sims, states):
mat      = len(returns[0,:])
llh      = np.zeros(sims)

# store variances and probabilities
vs       = np.zeros((sims, states, assets, assets))
ms       = np.zeros((sims, assets, states))
ps       = np.zeros((sims, states, states))

# Unimportant, but useful to initialise mu and var.
pStarRandom = np.random.uniform(size = mat * states).reshape(states, mat)

mu  = muFct(pStarRandom, returns, states)
var = varFct(pStarRandom, returns, mu, states)
p   = np.repeat(1.0 / states, states * states)

f   = fFct(returns, mu, var, states, assets)

a_r, a_s = aFct(mat, states, f, p)
b_r      = bFct(mat, states, f, p)

pStar    = pStarFct(mat, states, a_r, b_r)
pStarT   = pStarTFct(f, mat, states, a_r, a_s, b_r, p)

# 3. EM-loop until convergence (we loop sims amount of times)
for m in range(sims):
    # Reevaluate parameters given pStar    
    mu   = muFct(pStar, returns, states)
    var  = varFct(pStar, returns, mu, states)
    f    = fFct(returns, mu, var, states, assets)
    p    = pFct(pStarT, states)

    # New smoothed probabilities
    a_r, a_s = aFct(mat, states, f, p)
    b_r = bFct(mat, states, f, p)

    pStar  = pStarFct(mat, states, a_r, b_r)
    pStarT = pStarTFct(f, mat, states, a_r, a_s, b_r, p)
    
    # Compute the log-likelihood to maximise
    logLik = logLikFct(returns, mu, var, p, pStar, pStarT, f)

    # Save parameters for later plotting (redundant wrt optimisation)
    ms[m]  = mu
    vs[m]  = var
    ps[m]  = p.reshape(3,3)
    llh[m] = logLik


# test = runEstimation(y, sims, states)

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

if states == 2:
    pltm.plotDuo(range(sims), vs[0,:], vs[1,:], 'Var_1', 'Var_2', 'Trials', 'Variance')
    pltm.plotDuo(range(sims), ms[0,:], ms[1,:], 'Mu_1', 'Mu_2', 'Trials', 'Mean return')
    pltm.plotDuo(range(sims), ps[0,:], ps[3,:], 'p11', 'p22', 'Trials', 'Probability')
elif states == 3:
    # vs: vs[m, s, asset, asset] -> covariance for vs[m, s, 0, 1] or vs[m, s, 1, 0]
    pltm.plotTri(range(sims), vs[:,0,0,0], vs[:, 1, 0, 0], vs[:, 2, 0, 0], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[0]))
    pltm.plotTri(range(sims), vs[:,0,1,1], vs[:, 1, 1, 1], vs[:, 2, 1, 1], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[1]))
    pltm.plotTri(range(sims), vs[:,0,2,2], vs[:, 1, 2, 2], vs[:, 2, 2, 2], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[2]))
    pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 3, 3], vs[:, 2, 3, 3], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[3]))
    pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 4, 4], vs[:, 2, 4, 4], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[4]))
    pltm.plotTri(range(sims), ms[:, 0, 0], ms[:, 0, 1], ms[:, 0, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[0]))
    pltm.plotTri(range(sims), ms[:, 1, 0], ms[:, 1, 1], ms[:, 1, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[1]))
    pltm.plotTri(range(sims), ms[:, 2, 0], ms[:, 2, 1], ms[:, 2, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[2]))
    pltm.plotTri(range(sims), ms[:, 3, 0], ms[:, 3, 1], ms[:, 3, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[3]))
    pltm.plotTri(range(sims), ms[:, 4, 0], ms[:, 4, 1], ms[:, 4, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[4]))
    pltm.plotTri(range(sims), ps[:,0,0], ps[:, 1, 1], ps[:, 2, 2], 'Trials', 'p11', 'p22', 'p33', 'Probability')
    pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
elif states == 4:
    pltm.plotQuad(range(sims), vs[0,:], vs[1,:], vs[2,:], vs[3,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Var_4', 'Variance')
    pltm.plotQuad(range(sims), ms[0,:], ms[1,:], ms[2,:], ms[3,:], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mu_4', 'Mean return')
    pltm.plotQuad(range(sims), ps[0,:], ps[5,:], ps[10,:], ps[15,:], 'Trials', 'p11', 'p22', 'p33', 'p44', 'Probability')
    pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[3,:], xLab = 'Time', yLab = 'p4', title = 'Smoothed State Probabilities')

pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

