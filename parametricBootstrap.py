# -*- coding: utf-8 -*-
"""
Date:    February 20th, 2019
Authors: Kristian Strand and William Gram
Subject: Parameter estimation using the bootstrap method

Description:
Applying the bootstrap method from page 58 of Hidden Markov
Models for Time Series - an introduction using R by Zucchini,
MacDonald and Langrock.
"""

import genData as gd
import MSreturnSim as rs
from matplotlib import pyplot as plt
import plotsModule as pltm
import EM_NM_EX as em
import numpy as np
import pandas as pd

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ============================================= #
# ===== Idiosyncratic removal ================= #
# ============================================= #

"""
# ===== Monthly Absolute ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
monthlyRets = monthlyRets.drop(['S&P 500'], axis = 1)
colNames = monthlyRets.columns
assets = len(colNames)
returns = np.array(monthlyRets.T)
"""

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
assets = len(colNames)
returns = np.array(excessMRets.T)

# ============================================= #
# ===== Parameter initialisation ============== #
# ============================================= #

emSims = 100 # [EM]-algorithm simulations
bsSims = 30  # [b]ootstrap[S]ims
states = 3
mat    = len(returns[0,:])
probs  = np.repeat(1.0 / states, states * states).reshape(states, states)

m   = np.zeros((bsSims, assets, states))
v   = np.zeros((bsSims, states, assets, assets))
p   = np.zeros((bsSims, states, states))
l   = np.zeros((bsSims))
pSt = np.zeros((bsSims, states, mat))
pSt[0] = np.random.uniform(size = states * mat).reshape(states, mat)

ms, vs, ps, llh, pStar, pStarT = em.EM(returns, emSims, mat, states, assets, probs, pSt[0])

m[0]   = ms[emSims - 1]
v[0]   = vs[emSims - 1]
p[0]   = ps[emSims - 1]
l[0]   = llh[emSims - 1]
pSt[0] = pStar

u = np.random.uniform(0, 1, size = mat * bsSims).reshape(mat, bsSims)

for r in range(1,bsSims):
    # Starting regime
    startReg = np.argmax(pStar[:,0]) + 1 #technicality due to indexing.
    mu    = m[r-1]
    cov   = v[r-1]
    probs = p[r-1]

    simReturns = rs.returnSim3(states, assets, startReg, mu, cov, probs, mat, u[:,r])

    ms, vs, ps, llh, pStar, pStarT = em.EM(simReturns, emSims, mat, states, assets, probs, pSt[0])

    m[r]   = ms[emSims - 1]
    v[r]   = vs[emSims - 1]
    p[r]   = ps[emSims - 1]
    l[r]   = llh[emSims - 1]
    pSt[r] = pStar

stateTitle = ['State '+i for i in map(str,range(1, states + 1))]
volTitle   = ['Volatility, state ' + i for i in map(str, range(1, states + 1))]
retTitle   = ['Return, state ' + i for i in map(str, range(1, states + 1))]

# Log-likelihood convergence plot
pltm.plotUno(range(bsSims), l, yLab = 'log-likelihood value')

# Plot stay probabilities for each regime 1,2,...,S
idx  = np.zeros((bsSims, states)).astype(int)
for i in range(bsSims):
    idx[i,0]  = int(np.argmax(np.diag(p[i,:,:])))
    idx[i,2]  = int(np.argmin(np.diag(p[i,:,:])))
    idx[i,1]  = int(np.where(np.diag(p[i,:,:]) == np.median(np.diag(p[i,:,:])))[0])

# Probabilities being plotted
for i in range(states):
    plt.plot(np.diag(p[:,idx[:,i],idx[:,i]]))
plt.show()

# Plotting of returns
newMu = np.zeros((bsSims, states))
for j in range(bsSims):
    newMu[j,:] = m[j,0,idx[j,:]]

# WIP: plot confidence bands around e.g. mu
test  = np.zeros((bsSims,states,states))
for i in range(states):
    test[:,0,i] = np.quantile(newMu[:,i], 0.25)
    test[:,1,i] = np.mean(newMu[:,i])
    test[:,2,i] = np.quantile(newMu[:,i], 0.75)

for i in range(states):
    plt.plot(newMu[:,i])
    plt.plot(test[:,1,i])
plt.show()

idx