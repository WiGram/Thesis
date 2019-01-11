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

# ===== Gathered parameters =================== #

muS = np.array([-0.0510, 0.0069,  0.0116, 0.0519])
muB = np.array([-0.0131, 0.0009, -0.0023, 0.0136])
rf  = 0.053

eps =  np.array([[[0.1625 ** 2,              -0.4060 * 0.1625 * 0.0902],
                  [-0.4060 * 0.1625 * 0.0902, 0.0902 ** 2]],
                 [[0.1118 ** 2,               0.2043 * 0.1118 * 0.0688],
                  [0.2043 * 0.1118 * 0.0688,  0.0688 ** 2]],
                 [[0.1133 ** 2,               0.1521 * 0.1133 * 0.0261],
                  [0.1521 * 0.1133 * 0.0261,  0.0261 ** 2]],
                 [[0.1429 ** 2,               0.3692 * 0.1429 * 0.1000],
                  [0.3692 * 0.1429 * 0.1000,  0.1000 ** 2]]])

probs = np.array([[0.4940, 0.0001, 0.02409, 0.4818],
                  [0.0483, 0.8529, 0.0307,  0.0682],
                  [0.0439, 0.0701, 0.8822,  0.0038],
                  [0.0616, 0.1722, 0.0827,  0.6836]])

# ===== Models ================================ #
" Markov switching SV model "

mat = 6
time = np.arange(0,mat)

# --------------------------------------------- #
# Regime estimation

# Starting from state 1

mm, stocks, bonds = 0, 0, 0
remainder = []
M = 10
N = 500
for m in range(M):
    m
    u = np.random.uniform(0, 1, size = mat)
    for n in range(N):
        n
        state_ms = np.repeat(3, mat)
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

        # ============================================= #

        # Return estimation
        mu  = np.array([muS, muB])
        cov1 = eps[0]

        eps1 = eps[0]
        z = np.random.normal(0,1, size = mat)

        retS, retB = (state_ms == 1) * np.random.multivariate_normal(mu[:,0], eps[0], mat).T + \
                        (state_ms == 2) * np.random.multivariate_normal(mu[:,1], eps[1], mat).T + \
                        (state_ms == 3) * np.random.multivariate_normal(mu[:,2], eps[2], mat).T + \
                        (state_ms == 4) * np.random.multivariate_normal(mu[:,3], eps[3], mat).T

        # sp.std(retS), sp.std(retB)
        # sp.mean(retS), sp.mean(retB)

        # ((1. + sp.mean(retS)) ** 12 - 1.) * 100
        # ((1. + sp.mean(retB)) ** 12 - 1.) * 100

        # pltm.plotTri(time, retB, retS, state_ms, 'time', 'Bonds', 'Stocks', 'State', 'returns (states)')

        wS = np.arange(0, 1.01, 0.01)
        for idx in range(len(wS)):
            wS[idx] = np.round(wS[idx], 2)
        wB = np.ones(len(wS) ** 2).reshape(len(wS), len(wS))

        for decimal in wS:
            idx = np.where(wS == decimal)[0][0]
            a = wS[idx] + 0.01 # DO NOT CHANGE !!!
            w = np.arange(0,np.round(1.02 - a,2), 0.01)
            wB[idx,:] = np.append(w, np.repeat(0., len(wS) - len(w)))


        e = np.exp(sum(rf + retS)), np.exp(sum(rf + retB))

        gamma = 5.

        value = np.ones(len(wS) ** 2).reshape(len(wS), len(wS))
        for idx in range(len(wS)):
            s = wS[idx]
            for k in range(len(wS)):
                b = wB[idx, k]
                value[idx, k] = ( (1. - s - b) * np.exp(mat * rf) + s * e[0] + b * e[1]) ** (1. - gamma) / (1. - gamma)

        goLong = np.unravel_index(np.argmax(value, axis=None), value.shape)

        mm += (sum(goLong) == 0) * 1
        stocks += (goLong[0] == 100) * 1
        bonds  += (goLong[1] == 100) * 1

        if (sum(goLong) == 0) * 1 + (goLong[0] == 100) * 1 + (goLong[1] == 100) * 1 == 0:
            remainder.append(goLong) 
        

mmPct    = mm / (N * M)
stockPct = stocks / (N * M)
bondsPct = bonds / (N * M)

mmPct
stockPct
bondsPct
remainder

len(remainder)
stocks + mm + bonds + len(remainder) / 2