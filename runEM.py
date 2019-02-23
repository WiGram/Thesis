# -*- coding: utf-8 -*-
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
from matplotlib import pyplot as plt
import EM_NM_EX as em
import plotsEM as pem
import numpy as np
import pandas as pd

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

""" Monthly Absolute """
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


sims   = 200
states = 2
stateTitle = ['State '+i for i in map(str,range(1, states + 1))]
volTitle = ['Volatility, state' + i for i in map(str, range(1, states + 1))]
retTitle = ['Return, state' + i for i in map(str, range(1, states + 1))]
mat    = len(returns[0,:])
p      = np.repeat(1.0 / states, states * states).reshape(states, states)
pS     = np.random.uniform(size = states * mat).reshape(states, mat)

ms, vs, ps, llh, pStar, pStarT = em.EM(returns, sims, mat, states, assets, p, pS)
# pem.returnPlots(states, sims, vs, ms, ps, pStar, llh, colNames, rDates)

import plotsModule as pltm
pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

fig, axes = plt.subplots(nrows = 1, ncols = states, sharex = True, figsize = (15,6))
test = np.array([pStar[j,:] for j in range(states)])
for ax, title, y in zip(axes.flat, stateTitle, test):
    ax.plot(rDates, y)
    ax.set_title(title)
    ax.grid(False)
plt.show()

pltm.plotTri(range(sims), ps[:,0,0], ps[:, 1, 1], ps[:, 2, 2], 'Trials', 'p11', 'p22', 'p33', 'Probability')

for j, txt in zip(range(states), volTitle):
    fig, axes = plt.subplots(nrows = 3, ncols = 2, sharex = True, figsize = (15,15))
    fig.suptitle(txt, fontsize=16)

    test = np.array([vs[:,j,i,i] for i in range(assets)])
    for ax, title, y in zip(axes.flat, colNames, test):
        ax.plot(range(sims), y)
        ax.set_title(title)
        ax.grid(False)
    plt.show()

# sharex = True: the x-axis will be the same for all.. sharey = True is also possible
for j, txt in zip(range(states), retTitle):
    fig, axes = plt.subplots(nrows = 3, ncols = 2, sharex = True, figsize = (15,15))
    fig.suptitle(txt, fontsize=16)

    test = np.array([ms[:,i,j] for i in range(assets)])
    for ax, title, y in zip(axes.flat, colNames, test):
        ax.plot(range(sims), y)
        ax.set_title(title)
        ax.grid(False)
    plt.show()


