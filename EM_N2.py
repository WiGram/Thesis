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

# Output is f1, f2, ..., fN; var, mu must be arrays of parameters (i.e. for each state)
def muFct(pStar, returns, states):
    # mu = [sum(pStar[s, :] * returns[i,:]) / sum(pStar[s, :]) for s in range(states) for i in range(len(returns[:,0]))] # further inspection
    mu1 = [sum(pStar[s, :] * returns[0,:]) / sum(pStar[s, :]) for s in range(states)]
    mu2 = [sum(pStar[s, :] * returns[1,:]) / sum(pStar[s, :]) for s in range(states)]
    return np.array([mu1, mu2])

# Output: v1^2, v2^2, ..., vN^2
def varFct(pStar, returns, mu, states):
    var1 = [sum(pStar[s, :] * (returns[0,:] - mu[0,s]) ** 2) / sum(pStar[s, :]) for s in range(states)]
    var2 = [sum(pStar[s, :] * (returns[1,:] - mu[1,s]) ** 2) / sum(pStar[s, :]) for s in range(states)]
    cov  = [sum(pStar[s, :] * (returns[0,:] - mu[0,s]) * (returns[1,:] - mu[1,s])) / sum(pStar[s, :]) for s in range(states)]
    covm = [ np.array([[var1[s],cov[s]], [cov[s],var2[s]]]) for s in range(states)]
    return np.array(covm)

def fFct(returns, mu, covm, states):
    det             = np.linalg.det(covm) # returns [det1, det2, ..., detN]
    demeanedReturns = np.array([ np.array([ returns[0,:] - mu[0,s], 
                                            returns[1,:] - mu[1,s]  ]) for s in range(states)])
    f = [np.array([1 / np.sqrt( (2 * np.pi) ** 2 * det[s]) * np.exp(-0.5 * demeanedReturns[s,:,t].dot(np.linalg.inv(covm[s])).dot(demeanedReturns[s,:,t])) for s in range(states)]) for t in range(mat-1)]
    return np.array(f)

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
def logLikFct(mu, var, p, pStar, pStarT):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    a = sum([sum([np.log(p[s * states + i]) * sum(pStarT[s * states + i, 1:]) for i in range(states)]) for s in range(states)])
    b = sum([-0.5 * sum(pStar[s, :] * (np.log(var[s]) + (y -mu[s]) ** 2 / var[s])) for s in range(states)])
    return k + a + b

# ============================================= #
# ===== Start running the programme =========== #
# ============================================= #

# 0. Load S&P 500 data
# sp500 = pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx')
# d     = np.array(sp500['Date'][15096:], dtype = 'datetime64[D]')
# y     = np.array(sp500['log-ret_x100'][15096:]) # returns

sbl = ['^GSPC','AAPL']
bgn = '2010-01-01'
end = '2015-09-17'
src = 'yahoo'

# Returns apple first, S&P second..
close = web.DataReader(sbl, src, bgn, end)['Close']
# sp500 = web.DataReader(sbl, src, bgn, end)['Close'].to_frame().resample('MS').mean().round() # Monthly
prices = np.array(close)
d     = np.array(close.index, dtype = 'datetime64[D]')[1:]
mat   = len(prices[:,0])
y     = np.array([(np.log(prices[1:,i]) - np.log(prices[:mat-1,i])) * 100 for i in range(len(sbl))])

returns = y # will need for generating the dynamic function later..

states   = 3
assets   = len(sbl)
sims     = 1000

# 1. Set initial parameters
# def runEstimation(returns, sims, states):
mat      = len(returns[0,:])
llh      = np.zeros(sims)

# store variances and probabilities
vs       = np.zeros(states * sims).reshape(states, sims) # Needs to be fixed
ms       = np.zeros(states * sims).reshape(states, sims) # Needs to be fixed
ps       = np.zeros(states * states * sims).reshape(states * states, sims) # Needs to be fixed

# var won't work with e.g. np.ones(states), hence the "weird" construction
pStarRandom = np.random.uniform(size = mat * states).reshape(states, mat)

mu  = muFct(pStarRandom, returns, states)
var = varFct(pStarRandom, returns, mu, states)
p   = np.repeat(1.0 / states, states * states)

f       = fFct(returns, mu, var, states)

a_r, a_s = aFct(mat, states, f, p)
b_r      = bFct(mat, states, f, p)

pStar    = pStarFct(mat, states, a_r, b_r)
pStarT   = pStarTFct(mat, states, a_r, a_s, b_r, p)

# 3. EM-loop until convergence (we loop sims amount of times)
for m in range(sims):
    # Reevaluate parameters given pStar    
    mu   = muFct(pStar, returns, states)
    var  = varFct(pStar, returns, mu, states)
    f    = fFct(returns, mu, var, states)
    p    = pFct(pStarT, states)

    # New smoothed probabilities
    a_r, a_s = aFct(mat, states, f, p)
    b_r = bFct(mat, states, f, p)

    pStar  = pStarFct(mat, states, a_r, b_r)
    pStarT = pStarTFct(mat, states, a_r, a_s, b_r, p)
    
    # Compute the log-likelihood to maximise
    logLik = logLikFct(mu, var, p, pStar, pStarT)

    # Save parameters for later plotting (redundant wrt optimisation)
    ms[:, m]  = mu
    vs[:, m] = var
    ps[:, m] = p
    llh[m] = logLik

    # return np.array(vs, ms, ps, llh, pStar, pStarT)

# test = runEstimation(y, sims, states)

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

if states == 2:
    pltm.plotDuo(range(sims), vs[0,:], vs[1,:], 'Var_1', 'Var_2', 'Trials', 'Variance')
    pltm.plotDuo(range(sims), ms[0,:], ms[1,:], 'Mu_1', 'Mu_2', 'Trials', 'Mean return')
    pltm.plotDuo(range(sims), ps[0,:], ps[3,:], 'p11', 'p22', 'Trials', 'Probability')
elif states == 3:
    pltm.plotTri(range(sims), vs[0,:], vs[1,:], vs[2,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance')
    pltm.plotTri(range(sims), ms[0,:], ms[1,:], ms[2,:], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return')
    pltm.plotTri(range(sims), ps[0,:], ps[4,:], ps[8,:], 'Trials', 'p11', 'p22', 'p33', 'Probability')
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

# Generalise plots

pltm.plotUno(range(mat-1), f[:,0], yLab='density')