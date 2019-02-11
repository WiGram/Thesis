"""
Date:    February 6th, 2019
Authors: Kristian Strand and William Gram
Subject: Describing Bloomberg time series

Description:
This script is strictly exploratory, considering
monthly Bloomberg data. Time series are illustrated
and the first two moments are estimated. Data is gathered
by calling the genData.py script.
"""

import markowitzOpt as mpt
import genData as gd
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from scipy.optimize import minimize
from matplotlib import pyplot as plt
# import matplotlib.style as style
from numba import jit
from pandas_datareader import data as web
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')
np.set_printoptions(suppress = True)   # Disable scientific notation


prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# Plot index price series
prices.plot()
plt.show()

# Generate return correlation plots
"""
Arguably, true correlation should be measured by excess-returns,
arguing that all indices are affected by the risk free return level.
""" 
import seaborn as sns
from scipy import stats
g = sns.PairGrid(monthlyRets, vars = colNames)
g = g.map_diag(sns.distplot, fit = stats.norm)
g = g.map_offdiag(sns.scatterplot)
plt.show()

g = sns.PairGrid(excessMRets, vars = colNames)
g = g.map_diag(sns.distplot, fit = stats.norm)
# g = g.map_offdiag(sns.kdeplot, n_levels=6)
g = g.map_offdiag(sns.scatterplot)
plt.show()

long_df = monthlyRets.stack().reset_index(name = 'Returns')
long_df = long_df.rename(columns={'level_1':'Asset class',})
g = sns.FacetGrid(data = long_df, col = 'Asset class', col_wrap = 2, height = 4)
g.map(sns.distplot, 'Returns', fit = stats.norm)
plt.show()

# Time series plots of each return process
monthlyRets.plot(subplots = True, layout = (int(assets / 2), 2), figsize = (8,16))
plt.show()

# Moments and quartiles of return processes
summary = monthlyRets.describe().transpose()

# ============================================= #
# ===== Analysis of volatility ================ #
# ============================================= #

# Time series plot of each squared return process
monthlyVol.plot(subplots = True, layout = (int(assets / 2), 2), figsize = (8,16))
plt.show()

# ============================================= #
# ===== SR ==================================== #
# ============================================= #

list(summary)
l  = summary.index
m  = summary['mean'] * 12
s  = summary['std'] * np.sqrt(12)
SR = m / s

fig, ax = plt.subplots()
ax.scatter(s, m)
for i, txt in enumerate(l):
    ax.annotate(txt, (s[i], m[i]))
plt.grid()
plt.xlabel("Annualised volatility")
plt.ylabel("Annualised return")
plt.show()

# ============================================= #
# ===== Analysis of autocorrelation =========== #
# ============================================= #

def label(ax, string):
    ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                size=14, xycoords='axes fraction', textcoords='offset points')

fig, axes = plt.subplots(nrows=assets, figsize=(8, 12))
fig.tight_layout()

# Der antydes her en fejl. Dette ser ud til at være en falsk positiv - der er ingen fejl
for i in range(assets):
    pd.tools.plotting.autocorrelation_plot(monthlyRets.iloc[1:21,0], ax = axes[i])
    label(axes[i], colNames[i])
plt.show()

# ============================================= #
# ===== Analysis of optimal MPT portfolio ===== #
# ============================================= #

sims      = 50000
mptOutput = mpt.mptPortfolios(sims, monthlyRets, assets)
mpt.mptScatter(mptOutput['pfVol'], mptOutput['pfRet'],mptOutput['pfSR'],mptOutput['weights'],monthlyRets, n = 12)

list(mptOutput.keys())

"""
The end
"""