"""
Date:    February 10th, 2019
Authors: Kristian Strand and William Gram
Subject: Markowitz portfolio theory (MPF)

Description:
This script implements an algorithm to find
the optimal portfolio allocation through
Markowitz Portfolio Theory, i.e. optimising
the Sharpe ratio.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# Technicality on plotting layout
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')
np.set_printoptions(suppress = True)   # Disable scientific notation

# Optimisation functions
def neg_sharpe(weights, returns, n):
    return  get_ret_vol_sr(weights, returns, n)[2] * -1

# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

# Alternative approach to finding optimal weights
def get_ret_vol_sr(weights, returns, n):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    #
    1. Be observant as to "returns" being well-defined
    2. Be observant as to "n" being well-defined
    """
    weights = np.array(weights)
    ret = np.sum(returns.mean() * weights) * n
    vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * n, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def mptPortfolios(sims, returns, assets, frequency = 'm'):
    # Frequency must be either days or months
    if frequency == 'm':
        n = 12
    elif frequency == 'd':
        n = 252
    else:
        raise Exception("Use 'd' for 'days' or 'm' for 'months'")
    
    retMeans = returns.mean()
    retCov   = returns.cov()
    #
    weights  = np.random.random(size = (assets, sims))
    colSum   = np.sum(weights, axis = 0)
    weights /= colSum
    #
    pfRet = np.array([np.sum(retMeans * weights[:,i]) * n for i in range(sims)])
    pfVol = np.array([np.sqrt(np.dot(weights[:,i].T, np.dot(retCov * n, weights[:,i]))) for i in range(sims)])
    pfSR  = pfRet / pfVol
    #
    maxSR = pfSR.max()
    maxID = pfSR.argmax()
    maxRet = pfRet[maxID]
    maxVol = pfVol[maxID]
    maxWeights = weights[:,maxID]
    #
    return {'pfVol':pfVol,'pfRet':pfRet,'pfSR':pfSR,'weights':weights,'maxSR':maxSR,'maxRet':maxRet,'maxVol':maxVol,'maxWeights':maxWeights}

"""
TEST:
mpt = mptPortfolios(1000, excessMRets, assets)
pfVol = mpt['pfVol']
pfRet = mpt['pfRet']
pfSR = mpt['pfSR']
pfWeights = mpt['weights']
"""

def mptScatter(pfVol, pfRet, pfSR, weights, returns, n = 12,path=None):
    assets = len(weights[:,0])
    
    maxID = pfSR.argmax()
    maxRet = pfRet[maxID]
    maxVol = pfVol[maxID]
    
    frontier_y = np.linspace(pfRet.min(),pfRet.max(),100)
    
    def minimize_volatility(weights, returns, n):
        return  get_ret_vol_sr(weights, returns, n)[1]
    
    frontier_volatility = []
    
    # 0-1 bounds for each weight
    bounds = np.array([(0,1) for i in range(assets)])
    
    # Initial Guess (equal distribution)
    init_guess = np.repeat(1 / assets, assets)
    
    for possible_return in frontier_y:
        # function for return
        cons = ({'type':'eq','fun': check_sum},
                {'type':'eq','fun': lambda w: get_ret_vol_sr(w, returns, n)[0] - possible_return})
        
        result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons, args = (returns, n))
        
        frontier_volatility.append(result['fun'])
    
    plt.figure(figsize=(12,8))
    plt.scatter(pfVol, pfRet, c = pfSR, cmap = 'plasma')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    
    # Add red dot for max SR
    plt.scatter(maxVol,maxRet,c='red',s=50,edgecolors='black')
    
    # Add frontier line
    plt.plot(frontier_volatility,frontier_y,'g--',linewidth=3)
    
    if path != None:
        plt.savefig(path,bbox_inches = 'tight',pad_inches = 0)
    
    plt.show()

"""
TEST plot:
mptScatter(mpt['pfVol'], mpt['pfRet'],mpt['pfSR'],mpt['weights'],returns, n = 12)
"""