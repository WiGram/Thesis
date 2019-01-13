# -*- coding: utf-8 -*-
"""
Spyder Editor

Created on Sun Nov  4 08:57:28 2018

@author: William Gram
"""
# %reset -f
# %clear

import numpy as np
import likelihoodModule as llm
import plotsModule as pltm
import pandas as pd
import scipy as sp
import scipy.optimize as opt
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== parameters ============================ #
# ============================================= #

# Means
muS = np.array([-0.0510, 0.0069,  0.0116, 0.0519])
muB = np.array([-0.0131, 0.0009, -0.0023, 0.0136])
mu  = np.array([muS, muB])
rf  = 0.053 / 12

# Conditional covariances
cov =  np.array([[[0.1625 ** 2,              -0.4060 * 0.1625 * 0.0902],
                  [-0.4060 * 0.1625 * 0.0902, 0.0902 ** 2]],
                 [[0.1118 ** 2,               0.2043 * 0.1118 * 0.0688],
                  [0.2043 * 0.1118 * 0.0688,  0.0688 ** 2]],
                 [[0.1133 ** 2,               0.1521 * 0.1133 * 0.0261],
                  [0.1521 * 0.1133 * 0.0261,  0.0261 ** 2]],
                 [[0.1429 ** 2,               0.3692 * 0.1429 * 0.1000],
                  [0.3692 * 0.1429 * 0.1000,  0.1000 ** 2]]])

# Transition probabilities
probs = np.array([[0.4940, 0.0001, 0.02409, 0.4818],
                  [0.0483, 0.8529, 0.0307,  0.0682],
                  [0.0439, 0.0701, 0.8822,  0.0038],
                  [0.0616, 0.1722, 0.0827,  0.6836]])

# Grid search setup
wS = np.arange(0, 1.01, 0.01)
for idx in range(len(wS)):
    wS[idx] = np.round(wS[idx], 2)

wB = np.ones(len(wS) ** 2).reshape(len(wS), len(wS))
for decimal in wS:
    idx = np.where(wS == decimal)[0][0]
    a = 1.01 - wS[idx] - 0.001 # DO NOT CHANGE !!! (decimals problem with floats)
    w = np.arange(0,a, 0.01)
    wB[idx,:] = np.append(w, np.repeat(0., len(wS) - len(w)))

# Empty results container
value = np.zeros(len(wS) * len(wS)).reshape(len(wS), len(wS))

# ============================================= #
# ===== Models ================================ #
# ============================================= #

" Markov switching SV model "

# Starting from state 1
startReg = 3 # 1: Crash, 2: slowGrowth, 3: Bull, 4: Recovery
mat   = 6
gamma = 5
M = 100
N = 100

for m in range(M):
    u = np.random.uniform(0, 1, size = mat)
    state_ms = np.repeat(startReg, mat)
    for t in range(1,mat):
        if state_ms[t-1] == 1:
            state_ms[t] = \
                (u[t] <= probs[0,0]) * 1 + \
                (sum(probs[0,:1]) < u[t] <= sum(probs[0,:2])) * 2 + \
                (sum(probs[0,:2]) < u[t] <= sum(probs[0,:3])) * 3 + \
                (sum(probs[0,:3]) < u[t] <= 1) * 4
        elif state_ms[t-1] == 2:
            state_ms[t] = \
                (u[t] <= probs[1,0]) * 1 + \
                (sum(probs[1,:1]) < u[t] <= sum(probs[1,:2])) * 2 + \
                (sum(probs[1,:2]) < u[t] <= sum(probs[1,:3])) * 3 + \
                (sum(probs[1,:3]) < u[t] <= 1) * 4
        elif state_ms[t-1] == 3:
            state_ms[t] = \
                (u[t] <= probs[2,0]) * 1 + \
                (sum(probs[2,:1]) < u[t] <= sum(probs[2,:2])) * 2 + \
                (sum(probs[2,:2]) < u[t] <= sum(probs[2,:3])) * 3 + \
                (sum(probs[2,:3]) < u[t] <= 1) * 4
        else:
            state_ms[t] = \
                (u[t] <= probs[3,0]) * 1 + \
                (sum(probs[3,:1]) < u[t] <= sum(probs[3,:2])) * 2 + \
                (sum(probs[3,:2]) < u[t] <= sum(probs[3,:3])) * 3 + \
                (sum(probs[3,:3]) < u[t] <= 1) * 4

        lenOne = sum(state_ms == 1)
        lenTwo = sum(state_ms == 2)
        lenThr = sum(state_ms == 3)
        lenFou = sum(state_ms == 4)
    # ============================================= #

    for n in range(N):
        m, n
        # Return estimation
        rS, rB = np.ones(mat), np.ones(mat)

        rS[state_ms == 1], rB[state_ms == 1] = np.random.multivariate_normal(mu[:,0], cov[0], lenOne).T
        rS[state_ms == 2], rB[state_ms == 2] = np.random.multivariate_normal(mu[:,1], cov[1], lenTwo).T
        rS[state_ms == 3], rB[state_ms == 3] = np.random.multivariate_normal(mu[:,2], cov[2], lenThr).T
        rS[state_ms == 4], rB[state_ms == 4] = np.random.multivariate_normal(mu[:,3], cov[3], lenFou).T

        wR = wS * np.exp(sum(rf + rS)), wB * np.exp(sum(rf + rB))

        # Find out why we need to transpose
        value += ( (1. - wS - wB.T) * np.exp(mat * rf) + wR[0] + wR[1].T ).T ** (1 - gamma) / (1 - gamma)
    
goLong = np.unravel_index(np.argmax(value, axis=None), value.shape)

stocks = goLong[0] / 100
bonds = goLong[1] / 100
mm = 1 - stocks - bonds

stocks
bonds
mm