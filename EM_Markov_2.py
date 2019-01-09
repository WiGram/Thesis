# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import scipy.optimize as opt
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Initial functions ===================== #
# ============================================= #

def fFct(vol2, returns):
    return 1 / np.sqrt(2 * np.pi * vol2) * np.exp(-0.5 * returns ** 2 / vol2)

def s1Fct(pStar, returns):
    return sum(pStar[0, :] * returns ** 2) / sum(pStar[0, :])

def s2Fct(pStar, returns):
    return sum(pStar[1, :] * returns ** 2) / sum(pStar[1, :])

def p11Fct(pStarT):
    return sum(pStarT[0,:]) / sum(pStarT[0,:] + pStarT[1,:])

def p22Fct(pStarT):
    return sum(pStarT[3,:]) / sum(pStarT[2,:] + pStarT[3,:])

# A. Forward algorithm
def aFct(mat, states, f1, f2, p11, p22):
    p12, p21 = 1.0 - p11, 1.0 - p22

    a   = (f1[0] / mat, f2[0] / mat)
    a   = np.repeat(a, mat).reshape(states, mat)
    a_s = np.ones(mat)                               # a_scale
    a_r = np.ones(states * mat).reshape(states, mat) # a_rescale

    # t = 0
    a_s[0]    = sum(a[:,0])
    a_r[:, 0] = a[:,0] / a_s[0]

    # t in [1, T]
    for t in range(1, mat):
        a[0, t]   = f1[t] * sum([p11, p21] * a_r[:, t-1])
        a[1, t]   = f2[t] * sum([p12, p22] * a_r[:, t-1])
        a_s[t]    = sum(a[:, t])
        a_r[:, t] = a[:,t] / a_s[t]
        
    return a_r, a_s

# B. Backward algorithm
def bFct(mat, states, f1, f2, p11, p22):
    p12, p21 = 1.0 - p11, 1.0 - p22
    
    b   = np.ones(states * mat).reshape(states, mat)
    b_s = np.ones(mat)                               # b_scale
    b_r = np.ones(states * mat).reshape(states, mat) # b_rescale

    # t = T (= mat - 1)
    b_s[mat-1]      = sum(b[:, mat - 1])
    b_r[:, mat - 1] = b[:, mat - 1] / b_s[mat - 1]

    # t in [0, T - 1] (= mat - 2, stops at previous index, i.e. 0)
    for t in range(mat - 2, -1, -1):
        b[0, t]   = sum((b_r[0, t+1] * f1[t+1] * p11, b_r[1, t+1] * f2[t+1] * p12))
        b[1, t]   = sum((b_r[0, t+1] * f1[t+1] * p21, b_r[1, t+1] * f2[t+1] * p22))
        b_s[t]    = sum(b[:,t])
        b_r[:, t] = b[:, t] / b_s[t]

    return b_r

# C. Smoothed probabilities
def pStarFct(mat, states, a_r, b_r):
    pStar  = np.ones(states * mat).reshape(states, mat)
    denom = b_r[0,:] * a_r[0,:] + b_r[1,:] * a_r[1,:]
    for s in range(states):
        pStar[s, :] = (b_r[s, :] * a_r[s, :]) / denom
    
    return pStar

# D. Smoothed TRANSITION probabilities (these are joint probabilities, not conditional)
def pStarTFct(mat, states, a_r, a_s, b_r, p11, p22):
    pStarT = np.ones(states * states *mat).reshape(states * states, mat)
    p12, p21 = 1 - p11, 1 - p22

    denom = (b_r[0,:] * a_r[0,:] + b_r[1,:] * a_r[1,:]) * a_s
    pStarT[:, 0] = np.array([p11, p12, p21, p22]) / 2
    for t in range(1, mat):
        pStarT[0, t] = b_r[0, t] * f1[t] * p11 * a_r[0, t-1] / denom[t] #p*11
        pStarT[1, t] = b_r[1, t] * f2[t] * p12 * a_r[0, t-1] / denom[t] #p*12
        pStarT[2, t] = b_r[0, t] * f1[t] * p21 * a_r[1, t-1] / denom[t] #p*21
        pStarT[3, t] = b_r[1, t] * f2[t] * p22 * a_r[1, t-1] / denom[t] #p*22
    
    return pStarT

# E. Expected log-likelihood function to maximise
def logLikFct(s1, s2, p11, p22, pStar, pStarT):
    a = np.log(p11) * sum(pStarT[0, 1:]) + np.log(1 - p11) * sum(pStarT[1, 1:])
    b = np.log(p22) * sum(pStarT[3, 1:]) + np.log(1 - p22) * sum(pStarT[2, 1:])
    c = -0.5 * sum(pStar[0,:] * (np.log(s1) + y ** 2 / s1))
    d = -0.5 * sum(pStar[1,:] * (np.log(s2) + y ** 2 / s2))
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    return a + b + c + d + k

# ============================================= #
# ===== Start running the programme =========== #
# ============================================= #

# 0. Load S&P 500 data
sp500 = pd.DataFrame(pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx'))
y     = np.array(sp500['log-ret_x100'][15096:]) # returns

# 1. Set initial parameters

mat      = len(y)
states   = 2
sims     = 500
par      = np.zeros(4 * sims).reshape(4, sims)
llh      = np.zeros(sims)

s1, s2   = 1.0, 0.5
p11, p22 = 0.5, 0.5

f1       = fFct(s1, y)
f2       = fFct(s2, y)

a_r, a_s = aFct(mat, states, f1, f2, p11, p22)
b_r      = bFct(mat, states, f1, f2, p11, p22)

pStar    = pStarFct(mat, states, a_r, b_r)
pStarT   = pStarTFct(mat, states, a_r, a_s, b_r, p11, p22)
# Initial checks that all is OK
'OK: Minimum is weakly larger than 0' if min([min(pStar[i,:]) >= 0.0 for i in range(states)])                     else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStar[i,:]) <= 1.0 for i in range(states)])                     else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(pStar[0,:] + pStar[1,:], 10) == 1.0)                        else "Error: Probabilities don't sum to 1"
'OK: Minimum is weakly larger than 0' if min([min(pStarT[i,:]) >= 0.0 for i in range(states ** 2)])               else 'Error: Minimum is less than zero'
'OK: Maximum is weakly less than 1'   if min([max(pStarT[i,:]) <= 1.0 for i in range(states ** 2)])               else 'Error: Maximum is larger than 1'
'OK: Probabilities sum to 1'          if min(np.round(sum([pStarT[i,:] for i in range(states ** 2)]), 10) == 1.0) else "Error: Probabilities don't sum to 1"


# 3. EM-loop until convergence (we loop sims amount of times)
for m in range(sims):
    # Reevaluate parameters given pStar    
    s1  = s1Fct(pStar, y)
    s2  = s2Fct(pStar, y)
    
    # New densities
    f1     = fFct(s1, y)
    f2     = fFct(s2, y)
    
    # New Steady state probabilities
    p11 = p11Fct(pStarT)
    p22 = p22Fct(pStarT)
    
    # New smoothed probabilities
    a_r, a_s = aFct(mat, states, f1, f2, p11, p22)
    b_r = bFct(mat, states, f1, f2, p11, p22)

    pStar  = pStarFct(mat, states, a_r, b_r)
    pStarT = pStarTFct(mat, states, a_r, a_s, b_r, p11, p22)
    
    # Compute the log-likelihood to maximise
    logLik = logLikFct(s1, s2, p11, p22, pStar, pStarT)

    # Save parameters for later plotting (redundant wrt optimisation)
    par[0,m], par[1,m], par[2,m], par[3, m] = s1, s2, p11, p22
    llh[m] = logLik

pltm.plotDuo(range(sims), par[0,:], par[1,:], 'Sigma_h', 'Sigma_l', 'Time', 'Volatility')
pltm.plotDuo(range(sims), par[2,:], par[3,:], 'p11', 'p22', 'Time', 'Probability')
pltm.plotUno(range(sims), llh)

