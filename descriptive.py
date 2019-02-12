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

indexedPrices = prices / prices.iloc[0,:] * 100

# Plot index price series
indexedPrices.plot()
plt.savefig('/home/william/Dropbox/Thesis/Plots/Prices.pdf', bbox_inches = 'tight',
    pad_inches = 0)
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
plt.savefig('/home/william/Dropbox/Thesis/Plots/HistAndScatter.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

g = sns.PairGrid(excessMRets, vars = colNames)
g = g.map_diag(sns.distplot, fit = stats.norm)
# g = g.map_offdiag(sns.kdeplot, n_levels=6)
g = g.map_offdiag(sns.scatterplot)
plt.savefig('/home/william/Dropbox/Thesis/Plots/HistAndScatterExcess.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Density plots for each return process
long_df = monthlyRets.stack().reset_index(name = 'Returns')
long_df = long_df.rename(columns={'level_1':'Asset class',})
g = sns.FacetGrid(data = long_df, col = 'Asset class', col_wrap = 3, height = 6)
g.map(sns.distplot, 'Returns', fit = stats.norm)
plt.savefig('/home/william/Dropbox/Thesis/Plots/Densities.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Time series plots of each return process
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (15,15))

test = np.array([monthlyRets.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
plt.savefig('/home/william/Dropbox/Thesis/Plots/Returns.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Time serie plots of each excess return process
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (15,15))

test = np.array([excessMRets.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
plt.savefig('/home/william/Dropbox/Thesis/Plots/ExcessReturns.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Moments and quartiles of return processes
summaryRets = monthlyRets
summaryRets['Risk Free'] = rf

# Describe mean and standard deviation (columns 1 and 2 in .describe)
summary = summaryRets.describe().transpose().iloc[:,1:3]
summary['ann. mean'] = summary.iloc[:,0] * 12
summary['ann. std'] = summary.iloc[:,1] * np.sqrt(12)
summary['kurtosis'] = summaryRets.kurtosis()
summary['skewness'] = summaryRets.skew()

# Produce table in latex code
print(summary.round(4).to_latex())

# ============================================= #
# ===== Analysis of volatility ================ #
# ============================================= #

# Time series plot of each squared return process
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (15,15))

test = np.array([monthlyVol.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
plt.savefig('/home/william/Dropbox/Thesis/Plots/Vol.pdf', bbox_inches = 'tight',
    pad_inches = 0)
plt.show()


# monthlyVol.plot(subplots = True, layout = (int(assets / 2), 2), figsize = (8,16))
# plt.savefig('/home/william/Dropbox/Thesis/Plots/Vols.pdf', bbox_inches = 'tight',
#     pad_inches = 0)
# plt.show()

# ============================================= #
# ===== SR ==================================== #
# ============================================= #

list(summary)
l  = summary.iloc[:6,:].index
m  = summary.iloc[:6,2]
s  = summary.iloc[:6,3]
SR = m / s

fig, ax = plt.subplots()
ax.scatter(s, m)
for i, txt in enumerate(l):
    ax.annotate(txt, (s[i], m[i]))
plt.grid()
plt.xlabel("Annualised volatility")
plt.ylabel("Annualised return")
plt.savefig('/home/william/Dropbox/Thesis/Plots/Sharpes.pdf', bbox_inches = 'tight',
    pad_inches = 0)
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
    pd.tools.plotting.autocorrelation_plot(monthlyRets.iloc[1:21,i], ax = axes[i])
    label(axes[i], colNames[i])
plt.savefig('/home/william/Dropbox/Thesis/Plots/Autocorrelation.pdf', bbox_inches = 'tight',
    pad_inches = 0)
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