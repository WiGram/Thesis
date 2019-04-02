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
import statsmodels.tsa.api as smt
# import matplotlib.style as style
from numba import jit
from pandas_datareader import data as web
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')
np.set_printoptions(suppress = True)   # Disable scientific notation


prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()
prices = prices.drop(['S&P 500'], axis = 1)
monthlyVol = monthlyVol.drop(['S&P 500'], axis = 1)

excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns

assets = len(colNames)

indexedPrices = prices / prices.iloc[0,:] * 100

# Plot index price series
indexedPrices.plot()
path='/home/william/Dropbox/Thesis/Plots/Prices.png'
plt.savefig(path,bbox_inches='tight',pad_inches =0)
plt.show()

# Generate return correlation plots
"""
Arguably, true correlation should be measured by excess-returns,
arguing that all indices are affected by the risk free return level.
""" 
import seaborn as sns
from scipy import stats
g = sns.PairGrid(excessMRets, vars = colNames)
g = g.map_diag(sns.distplot, fit = stats.norm)
# g = g.map_offdiag(sns.kdeplot, n_levels=6)
g = g.map_offdiag(sns.scatterplot)
path='/home/william/Dropbox/Thesis/Overleaf/images/HistAndScatterExcess.png'
plt.savefig(path, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Density plots for each return process
long_df=excessMRets.stack().reset_index(name = 'Returns')
long_df=long_df.rename(columns={'level_1':'Asset class',})
g=sns.FacetGrid(data=long_df,col='Asset class',col_wrap=2,height=4,aspect=1.5)
g.map(sns.distplot, 'Returns', fit = stats.norm)
path='/home/william/Dropbox/Thesis/Overleaf/images/Densities.png'
plt.savefig(path, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Time series plots of each return process
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (14,16))
fig.text(0.06, 0.5, 'Returns (pct.)', va='center', rotation='vertical')

test = np.array([excessMRets.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
path='/home/william/Dropbox/Thesis/Overleaf/images/Returns.png'
plt.savefig(path, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Time serie plots of each excess return process
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (14,16))
fig.text(0.06,0.5,'Excess Returns (pct.)',va='center',rotation='vertical')

test = np.array([excessMRets.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
path='/home/william/Dropbox/Thesis/Overleaf/images/ExcessReturns.png'
plt.savefig(path, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()

# Moments and quartiles of return processes
summaryRets = excessMRets.copy()
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
                         figsize = (14,16))
fig.text(0.06, 0.5, 'Volatility (pct.)', va='center', rotation='vertical')

test = np.array([monthlyVol.iloc[:,i] for i in range(assets)])
for ax, title, y in zip(axes.flat, colNames, test):
    ax.plot(rDates, y)
    ax.legend(loc = 'lower right')
    ax.set_title(title)
    ax.grid(False)
path='/home/william/Dropbox/Thesis/Overleaf/images/Vol.png'
plt.savefig(path, bbox_inches='tight',
    pad_inches = 0)
plt.show()

# monthlyVol.plot(subplots=True,layout=(int(assets / 2), 2),figsize=(8,16))
# '/home/william/Dropbox/Thesis/Plots/Vols.pdf'
# plt.savefig(path,bbox_inches='tight',pad_inches=0)
# plt.show()

# ============================================= #
# ===== SR ==================================== #
# ============================================= #

list(summary)
l  = summary.index
m  = summary.iloc[:,2] # Annualised mean
s  = summary.iloc[:,3] # Annualised volatility
SR = m / s

fig, ax = plt.subplots()
ax.scatter(s, m)
for i, txt in enumerate(l):
    ax.annotate(txt, (s[i], m[i]))
plt.grid()
plt.xlabel("Annualised volatility")
plt.ylabel("Annualised excess return")
path='/home/william/Dropbox/Thesis/Overleaf/images/Sharpes.png'
plt.savefig(path,bbox_inches='tight',pad_inches=0)
plt.show()

# ============================================= #
# ===== Analysis of autocorrelation =========== #
# ============================================= #

# Labelling function
def label(ax, string):
    ax.annotate(string,(1, 1),xytext=(-8, -8), ha='right',va='top',size=14,
                xycoords='axes fraction',textcoords='offset points')

# Excess returns autocorrelation
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True, 
                         figsize = (14,16))
fig.tight_layout()
fig.subplots_adjust(hspace=0.15)

for i, ax, title in zip(range(assets), axes.flat, colNames):
    smt.graphics.plot_acf(excessMRets.iloc[:,i],lags=30,ax=ax,title=title)
path='/home/william/Dropbox/Thesis/Overleaf/images/Autocorrelation.png'
plt.savefig(path,bbox_inches='tight',pad_inches=0)
plt.show()

# Squared excess returns autocorrelation
fig, axes = plt.subplots(nrows = 3, 
                         ncols = 2, 
                         sharex = True, 
                         sharey = True,
                         figsize=(14, 16))
fig.tight_layout()
fig.subplots_adjust(hspace=0.15)

for i, ax, title in zip(range(assets), axes.flat, colNames):
    smt.graphics.plot_acf(abs(excessMRets.iloc[:,i]),lags=30,ax=ax,title=title)
path='/home/william/Dropbox/Thesis/Overleaf/images/AutocorrelationSquared.png'
plt.savefig(path,bbox_inches='tight',pad_inches = 0)
plt.show()


"""
# Checking if scipy.stats.kurtosis is Kurtosis or excess Kurtosis.
# It turns out the excess kurtosis is computed.

import scipy
test = np.array(excessMRets.iloc[:,0])
muTest = scipy.stats.tmean(test)
muNew = sum(test) / len(test)
sdTest = scipy.stats.tstd(test)

kurt = scipy.stats.kurtosis(test)
kurtTest=len(test)*sum( (test - muTest)**4 )/(sum((test - muTest)**2) )**2
"""

# ============================================= #
# ===== Analysis of optimal MPT portfolio ===== #
# ============================================= #

sims      = 50000
mptOutput = mpt.mptPortfolios(sims, excessMRets, assets)
# The arguments produced in mpt.mptPortfolios
list(mptOutput.keys())

path='/home/william/Dropbox/Thesis/Overleaf/images/pfSR.png'
mpt.mptScatter(mptOutput['pfVol'],mptOutput['pfRet'],mptOutput['pfSR'],
               mptOutput['weights'],excessMRets,n=12,path=path)

# Fundamental characteristics: highest SR = gradient
SR = mptOutput['maxSR']
mptOutput['maxRet'] / mptOutput['maxVol']

mptOutput['maxWeights']

# Generate x-axis
sd=np.linspace(start=0.0,stop=mptOutput['pfVol'].max(),num=sims)

# Generate Capital Market Line
mu = SR*sd

# Generate utility function
u = 0.1 # utility level (curve placement)
c_low = 0.5 # parameter to be defined
c_hi  = 1.5

# Find tangency volatility
(SR + np.sqrt(SR**2 - 2*c_low*u))/c_low
(SR - np.sqrt(SR**2 - 2*c_low*u))/c_low
(SR + np.sqrt(SR**2 - 2*c_hi*u))/c_hi
(SR - np.sqrt(SR**2 - 2*c_hi*u))/c_hi

# Guarantee that the two points become the same, by varying u
u_low = SR**2/(2*c_low)
sd_star_low = SR / c_low
mu_star_low = SR*sd_star_low

u_hi = SR**2/(2*c_hi)
sd_star_hi = SR / c_hi
mu_star_hi = SR*sd_star_hi

weight_low = mu_star_low / mptOutput['maxRet']
weight_hi  = mu_star_hi / mptOutput['maxRet']

utility_low = u_low + c_low / 2 * sd ** 2
utility_hi  = u_hi + c_hi / 2 * sd ** 2

(SR + np.sqrt(SR**2 - 2*c_low*u_low))/c_low
(SR + np.sqrt(SR**2 - 2*c_low*u_hi))/c_hi

plt.figure(figsize=(12,8))
plt.plot(sd,mu)
plt.plot(sd,utility_low)
plt.plot(sd,utility_hi)
plt.scatter(mptOutput['pfVol'],mptOutput['pfRet'],
            c=mptOutput['pfSR'],cmap='plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.scatter(sd_star_low,mu_star_low,color='red')
plt.annotate('Weight, low risk aversion: {}'.format(round(weight_low,2)),
    (sd_star_low,mu_star_low))
plt.scatter(sd_star_hi,mu_star_hi,color='red')
plt.annotate('Weight, high risk aversion: {}'.format(round(weight_hi,2)),
    (sd_star_hi,mu_star_hi))
plt.ylim(bottom=0,top=10)
plt.xlim(left=0,right=10)
plt.xlabel('Volatility')
plt.ylabel('Return')
path='/home/william/Dropbox/Thesis/Overleaf/images/pfAllocUtility.png'
plt.savefig(path,bbox_inches='tight',pad_inches=0)
plt.show()



"""
The end
"""
