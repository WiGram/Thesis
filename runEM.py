"""
Date:    February 11th, 2019
Authors: Kristian Strand and William Gram
Subject: Describing Bloomberg time series

Description:
This script runs the EM algorithm by calling the EM_NM_EX.py script
by generating data in the genData.py script. Finally the output is
plotted by calling the plotsEM.py script.
"""

import genData as gd
import EM_NM_EX as em
import emPlots as emp
import numpy as np
import pandas as pd

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

"""
# ===== Monthly Absolute ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
monthlyRets = monthlyRets.drop(['S&P 500'], axis = 1)
colNames =.columns
assets = len(colNames)
returns = np.array(monthlyRets.T)
"""

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
assets = len(colNames)
returns = np.array(excessMRets.T)

sims   = 200
states = 3
mat    = len(returns[0,:])
p      = np.repeat(1.0 / states, states * states).reshape(states, states)
pS     = np.random.uniform(size = states * mat).reshape(states, mat)

ms, vs, ps, llh, pStar, pStarT = em.multEM(returns, sims, mat, states, assets, p, pS)

m, v, p, l, pss, pst = em.uniEM(returns[0], sims, mat, states, p, pS)

params = ms[sims-1], vs[sims-1], ps[sims-1], pStar, pStarT
# Plot all
emp.emPlots(sims, states, assets, rDates, colNames, llh, ps, vs, ms, pStar)
emp.emUniPlots(sims, states, rDates, colNames, l, p, v, m, pss)

from llhFct import llhFct

llhFct(params, returns)

import numdifftools as nd

jac_fct = nd.Jacobian(llhFct)

jac_fct(params, args = returns)