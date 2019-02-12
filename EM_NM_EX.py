# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import plotsModule as pltm
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from numba import jit
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Initial functions ===================== #
# ============================================= #

"""
Terminology

y =  returns
pS = pStar    # Smoothed probabilities
pST = pStarT  # Smoothed Transition probabilities
S = states
A = assets
T = mat       # length of time series

"""

# Output is f1, f2, ..., fN; var, mu must be arrays of parameters (i.e. for each state)
@jit(nopython = True)
def muFct(pS, y, S, A):
    
    def meanFct(p, y):
        return np.sum(p * y) / np.sum(p)
    
    mu  = [[meanFct(pS[s,:], y[i,:]) for s in range(S)] for i in range(A)]
    return np.array(mu)

# Output: v1^2, v2^2, ..., vN^2
@jit(nopython = True)
def covFct(p, y1, y2, mu1, mu2):
    return np.sum(p * (y1 - mu1) * (y2 - mu2)) / np.sum(p)

def varFct(pS, y, mu, S, A):   
    covm = [[covFct(pS[s, :], y[i,:], y[j,:], mu[i,s], mu[j,s]) for i in range(A) for j in range(A)] for s in range(S)]
    return np.array(covm).reshape(S,A,A)

@jit(nopython = True)
def density(d, dR, covm, A):
        return 1 / np.sqrt( (2 * np.pi) ** A * d) * np.exp(-0.5 * dR.dot(np.linalg.inv(covm)).dot(dR))

def fFct(y, mu, covm, S, A, T):
    """
    d:   determinant
    dR:  demeaned returns
    """
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.array([[y[i,:] - mu[i, s] for i in range(A)] for s in range(S)])

    f  = [[density(d[s], dR[s,:,t], covm[s], A) for s in range(S)] for t in range(T)]
    return np.array(f)

# Output: p11, p12, ..., p1N, p21, p22, ..., p2N, ..., pN1, pN2, ..., pNN
# @jit
def pFct(pST, S):
    """
    den: denominator
    """
    den = [np.sum([pST[s * S + i,:] for i in range(S)]) for s in range(S)]
    p   = [np.sum(pST[s * S + i,:]) / den[s] for s in range(S) for i in range(S)]
    return np.array(p)

# A. Forward algorithm
# @jit(nopython = True)
def aFct(T, S, f, p):
    """
    For time t = 0
    a  = v * f, where v := 1 / states; repeat to initialise, reshape to size S x T
    aS = a scaled
    aR = a rescaled
    Initialisation follows below.
    """

    """
    a  = np.ones(S * T).reshape(S, T)
    aS = np.ones(T)
    aR = np.ones(S * T).reshape(S, T)

    a[:,0]  = [f[s,0] / S for s in range(S)]
    aS[0]   = np.sum(a[:,0])
    aR[:,0] = a[:,0] / aS[0]
    """

    a  = np.repeat([f[s,0] / S for s in range(S)], T).reshape(S, T)
    aS = np.repeat(np.sum(a[:,0]), T)
    aR = np.repeat(a[:,0] / aS[0], T).reshape(S, T)

    """
    Fill out for t in [1, T]
    """
    for t in range(1, T):
        a[:, t]  = [f[t,s] * np.sum([p[i * S + s] * aR[i, t-1] for i in range(S)]) for s in range(S)]
        aS[t]    = np.sum(a[:, t])
        aR[:, t] = a[:,t] / aS[t]

    return np.array(aR), np.array(aS)

# B. Backward algorithm
# @jit(nopython = True)
def bFct(T, S, f, p):
    """
    bR = rescaled b
    Algorithm is backwards recursive
    """
    b  = np.ones(S * T).reshape(S, T)
    bR = np.repeat(b[:,T-1] / np.sum(b[:, T - 1]), T).reshape(S, T)

    """
    Fill out for t in [T-1: 0]
    range(Start, end, step) <- recall ends at index prior to end.
    """
    for t in range(T - 2, -1, -1):
        b[:, t]   = [np.sum([bR[i, t+1] * f[t+1,i] * p[s * S + i] for i in range(S)]) for s in range(S)]
        bR[:, t] = b[:, t] / np.sum(b[:,t])

    return np.array(bR)

# Output (smoothed) p1, p2, ..., pN
#@jit
def pStarFct(T, S, aR, bR):
    """
    den: denominator
    """
    den   = sum([bR[s, :] * aR[s, :] for s in range(S)])
    pStar = [bR[s, :] * aR[s,:] / den for s in range(S)]
    return np.array(pStar)

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
#@jit
def pStarTFct(f, T, S, aR, aS, bR, p):
    """
    den: denominator
    """
    den   = sum([bR[s, :] * aR[s, :] for s in range(S)]) * aS
    pStarT = np.repeat(p / S, T).reshape(S * S, T)

    pStarT[:, 1:] = [bR[s, 1:] * f[1:,s] * p[i * S + s] * aR[i, :T - 1] / den[1:] for i in range(S) for s in range(S)]
    return np.array(pStarT)

# E. Expected log-likelihood function to maximise
#@jit
def logLikFct(y, f, p, pS, pST, S):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    a = sum([sum([np.log(p[s * S + i]) * sum(pST[s * S + i, 1:]) for i in range(S)]) for s in range(S)])
    b = sum([-0.5 * sum(pS[s, :] * f[:, s]) for s in range(S)])
    return k + a + b

# ============================================= #
# ===== EM Algorithm ========================== #
# ============================================= #

#@jit(nopython=True)
def EM(returns, sims, mat, states, assets, p, pS):
    llh      = np.zeros(sims)

    # store variances and probabilities
    vs       = np.zeros((sims, states, assets, assets))
    ms       = np.zeros((sims, assets, states))
    ps       = np.zeros((sims, states, states))

    # pS is a formality that helps initialise mu, var
    mu  = muFct(pS, returns, states, assets)
    var = varFct(pS, returns, mu, states, assets)

    f   = fFct(returns, mu, var, states, assets, mat)

    a_r, a_s = aFct(mat, states, f, p)
    b_r      = bFct(mat, states, f, p)

    pStar    = pStarFct(mat, states, a_r, b_r)
    pStarT   = pStarTFct(f, mat, states, a_r, a_s, b_r, p)

    # 3. EM-loop until convergence (we loop sims amount of times)
    for m in range(sims):
        # Reevaluate parameters given pStar
        mu   = muFct(pStar, returns, states, assets)
        var  = varFct(pStar, returns, mu, states, assets)
        f    = fFct(returns, mu, var, states, assets, mat)
        p    = pFct(pStarT, states)

        # New smoothed probabilities
        a_r, a_s = aFct(mat, states, f, p)
        b_r = bFct(mat, states, f, p)

        pStar  = pStarFct(mat, states, a_r, b_r)
        pStarT = pStarTFct(f, mat, states, a_r, a_s, b_r, p)
        
        # Compute the log-likelihood to maximise
        logLik = logLikFct(returns, f, p, pStar, pStarT, states)

        # Save parameters for later plotting (redundant wrt optimisation)
        ms[m]   = mu
        vs[m]   = np.sqrt(var)
        ps[m]   = p.reshape(states,states)
        llh[m]  = logLik
    
    return ms, vs, ps, llh, pStar, pStarT

# ============================================= #
# ===== Start running the programme =========== #
# ============================================= #

# 0. Load S&P 500 data
# sp500 = pd.read_excel('C:/Users/wigr11ab/Dropbox/KU/K3/FE/Exercises/SP500.xlsx')
# d     = np.array(sp500['Date'][15096:], dtype = 'datetime64[D]')
# y     = np.array(sp500['log-ret_x100'][15096:]) # returns

"""
-------------------------------------------------------------------------------
Example of an application:
-------------------------------------------------------------------------------

sbl = ['AAPL','DJP','HYG','VBMFX','^GSPC'] # Apply alphabetical order.
bgn = '2010-01-01'
end = '2015-09-17'
src = 'yahoo'

#test = web.DataReader('DGS10', 'fred', bgn, end)

close = web.DataReader(sbl, src, bgn, end)['Close']
# sp500 = web.DataReader(sbl, src, bgn, end)['Close'].to_frame().resample('MS').mean().round() # Monthly
prices  = np.array(close)
d       = np.array(close.index, dtype = 'datetime64[D]')[1:]
mat     = len(prices[:,0])
returns = np.array([(np.log(prices[1:,i]) - np.log(prices[:mat-1,i])) * 100 for i in range(len(sbl))])

sims   = 50
states = 3 
assets = len(sbl)
mat    = len(returns[0,:])
p      = np.repeat(1.0 / states, states * states)
pS     = np.random.uniform(size = states * mat).reshape(states, mat)

# ms, vs, ps, llh, pStar, pStarT = EM(returns, sims, mat, states, assets, p, pS)

# test = runEstimation(y, sims, states)

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

if states == 2:
    pltm.plotDuo(range(sims), vs[0,:], vs[1,:], 'Var_1', 'Var_2', 'Trials', 'Variance')
    pltm.plotDuo(range(sims), ms[0,:], ms[1,:], 'Mu_1', 'Mu_2', 'Trials', 'Mean return')
    pltm.plotDuo(range(sims), ps[0,:], ps[3,:], 'p11', 'p22', 'Trials', 'Probability')
elif states == 3:
    # vs: vs[m, s, asset, asset] -> covariance for vs[m, s, 0, 1] or vs[m, s, 1, 0]
    pltm.plotTri(range(sims), vs[:,0,0,0], vs[:, 1, 0, 0], vs[:, 2, 0, 0], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[0]))
    pltm.plotTri(range(sims), vs[:,0,1,1], vs[:, 1, 1, 1], vs[:, 2, 1, 1], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[1]))
    pltm.plotTri(range(sims), vs[:,0,2,2], vs[:, 1, 2, 2], vs[:, 2, 2, 2], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[2]))
    pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 3, 3], vs[:, 2, 3, 3], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[3]))
    pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 4, 4], vs[:, 2, 4, 4], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % sbl[4]))
    pltm.plotTri(range(sims), ms[:, 0, 0], ms[:, 0, 1], ms[:, 0, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[0]))
    pltm.plotTri(range(sims), ms[:, 1, 0], ms[:, 1, 1], ms[:, 1, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[1]))
    pltm.plotTri(range(sims), ms[:, 2, 0], ms[:, 2, 1], ms[:, 2, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[2]))
    pltm.plotTri(range(sims), ms[:, 3, 0], ms[:, 3, 1], ms[:, 3, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[3]))
    pltm.plotTri(range(sims), ms[:, 4, 0], ms[:, 4, 1], ms[:, 4, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % sbl[4]))
    # pltm.plotTri(range(sims), arS[:, 0, 0], arS[:, 0, 1], arS[:, 0, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % sbl[0]))
    # pltm.plotTri(range(sims), arS[:, 1, 0], arS[:, 1, 1], arS[:, 1, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % sbl[1]))
    # pltm.plotTri(range(sims), arS[:, 2, 0], arS[:, 2, 1], arS[:, 2, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % sbl[2]))
    # pltm.plotTri(range(sims), arS[:, 3, 0], arS[:, 3, 1], arS[:, 3, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % sbl[3]))
    # pltm.plotTri(range(sims), arS[:, 4, 0], arS[:, 4, 1], arS[:, 4, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % sbl[4]))
    # pltm.plotTri(range(sims), avgS[:, 0, 0], avgS[:, 0, 1], avgS[:, 0, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % sbl[0]))
    # pltm.plotTri(range(sims), avgS[:, 1, 0], avgS[:, 1, 1], avgS[:, 1, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % sbl[1]))
    # pltm.plotTri(range(sims), avgS[:, 2, 0], avgS[:, 2, 1], avgS[:, 2, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % sbl[2]))
    # pltm.plotTri(range(sims), avgS[:, 3, 0], avgS[:, 3, 1], avgS[:, 3, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % sbl[3]))
    # pltm.plotTri(range(sims), avgS[:, 4, 0], avgS[:, 4, 1], avgS[:, 4, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % sbl[4]))
    pltm.plotTri(range(sims), ps[:,0,0], ps[:, 1, 1], ps[:, 2, 2], 'Trials', 'p11', 'p22', 'p33', 'Probability')
    pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
elif states == 4:
    pltm.plotQuad(range(sims), vs[0,:], vs[1,:], vs[2,:], vs[3,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Var_4', 'Variance')
    pltm.plotQuad(range(sims), ms[0,:], ms[1,:], ms[2,:], ms[3,:], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mu_4', 'Mean return')
    pltm.plotQuad(range(sims), ps[0,:], ps[5,:], ps[10,:], ps[15,:], 'Trials', 'p11', 'p22', 'p33', 'p44', 'Probability')
    pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
    pltm.plotUno(d, pStar[3,:], xLab = 'Time', yLab = 'p4', title = 'Smoothed State Probabilities')

pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

"""