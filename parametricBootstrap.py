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
import EM_NM_EX as em
import plotsEM as pem
import numpy as np
import pandas as pd

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ============================================= #
# ===== Idiosyncratic removal ================= #
# ============================================= #

# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
monthlyRets = monthlyRets.drop(['S&P 500'], axis = 1)
#test = monthlyRets.drop(['Commodities', 'S&P 500'], axis = 1)
colNames = monthlyRets.columns
#colNames.remove('Commodities')
assets = len(colNames)
returns = np.array(monthlyRets.T)
#returns = np.array(excessMRets.T)

""" Monthly excess """
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
#test = monthlyRets.drop(['Commodities', 'S&P 500'], axis = 1)
colNames = excessMRets.columns
#colNames.remove('Commodities')
assets = len(colNames)
returns = np.array(excessMRets.T)
#returns = np.array(excessMRets.T)

# ============================================= #
# ===== Parameter initialisation ============== #
# ============================================= #

emSims = 100 # [EM]-algorithm simulations
bsSims = 30   # [b]ootstrap[S]ims
states = 3
assets = 5
mat    = len(returns[0,:])
probs  = np.repeat(1.0 / states, states * states)
pS     = np.random.uniform(size = states * mat).reshape(states, mat)

m   = np.zeros((assets, states, bsSims))
v   = np.zeros((states, assets, assets, bsSims))
p   = np.zeros((states, states, bsSims))
l   = np.zeros(bsSims)

ms, vs, ps, llh, pStar, pStarT = em.EM(returns, emSims, mat, states, assets, probs, pS)

m[:,:, 0]  = ms[emSims - 1, :,:]
v[:,:,:,0] = vs[emSims - 1, :,:,:]
p[:,:,0]   = ps[emSims - 1, :,:]
l[0]       = llh[emSims - 1]

u = np.random.uniform(0, 1, size = mat * bsSims).reshape(mat, bsSims)

for r in range(1,bsSims):
    # Starting regime
    startReg = np.argmax(pStar[:,0]) + 1 #technicality due to indexing.
    mu    = m[:,:,r-1]
    cov   = v[:,:,:,r-1].reshape(states, assets, assets)
    probs = p[:,:,r-1]

    simReturns = rs.returnSim3(states, assets, startReg, mu, cov, probs, mat, u[:,r])

    ms, vs, ps, llh, pStar, pStarT = em.EM(simReturns, emSims, mat, states, assets, p[:,:,r-1].reshape(3 * 3), pS)

    m[:,:, r]  = ms[emSims - 1, :,:]
    v[:,:,:,r] = vs[emSims - 1, :,:,:]
    p[:,:,r]   = ps[emSims - 1, :,:]
    l[r]       = llh[emSims - 1]










"""
for r in range(1,bsSims):
    probs = ps[emSims - 1, :,:]
    ms, vs, ps, llh, pStar, pStarT = em.EM(returns, emSims, mat, states, assets, probs.reshape(3 * 3), pS)

    m[:,:, r]  = ms[emSims - 1, :,:]
    v[:,:,:,r] = vs[emSims - 1, :,:,:]
    p[:,:,r]   = ps[emSims - 1, :,:]
    l[r]       = llh[emSims - 1]

    # Starting regime
    startReg = np.argmax(pStar[:,0]) + 1 #technicality due to indexing.
    mu    = m[:,:,r-1]
    cov   = v[:,:,:,r-1].reshape(states, assets, assets)
    probs = p[:,:,r-1]




    simReturns = rs.returnSim3(states, assets, startReg, mu, cov, probs, mat, u[:,r])

    ms, vs, ps, llh, pStar, pStarT = em.EM(simReturns, emSims, mat, states, assets, p[:,:,r-1].reshape(3 * 3), pS)

    m[:,:, r]  = ms[emSims - 1, :,:]
    v[:,:,:,r] = vs[emSims - 1, :,:,:]
    p[:,:,r]   = ps[emSims - 1, :,:]
    l[r]       = llh[emSims - 1]
"""
