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

    def meanFct(pS, y):
        return np.sum(pS * y) / np.sum(pS)

    mu = np.zeros((A,S))

    for a in range(A):
        for s in range(S):
            mu[a,s] = meanFct(pS[s,:], y[a,:])
    
    return mu

# Output: v1^2, v2^2, ..., vN^2
@jit(nopython = True)
def covFct(pS, y1, y2, mu1, mu2):
    return np.sum(pS * (y1 - mu1) * (y2 - mu2)) / np.sum(pS)

@jit(nopython = True)
def varFct(pS, y, mu, S, A):

    covm = np.zeros((S,A,A))
    for s in range(S):
        for i in range(A):
            for j in range(A):
                covm[s, i, j] = covFct(pS[s,:], y[i,:],y[j,:],mu[i,s],mu[j,s])

    return covm

@jit(nopython = True)
def density(d, dR, covm, A):
        return 1 / np.sqrt( (2 * np.pi) ** A * d) * np.exp(-0.5 * dR.dot(np.linalg.inv(covm)).dot(dR))

@jit
def fFct(y, mu, covm, S, A, T):
    """
    d:   determinant
    dR:  demeaned returns
    """
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((S,A,T))
    for s in range(S):
        for a in range(A):
            dR[s,a,:] = y[a,:] - mu[a,s]

    f  = np.zeros((S, T))
    for s in range(S):
        for t in range(T):
            f[s, t] = density(d[s], dR[s, :, t], covm[s], A)

    return f

# Output: p11, p12, ..., p1N, p21, p22, ..., p2N, ..., pN1, pN2, ..., pNN
@jit(nopython = True)
def pFct(pST, S):
    """
    den: denominator
    """
    p = np.zeros((S, S))
    den = np.sum(np.sum(pST[:,:,1:], axis = 2), axis = 0)
    for i in range(S):
        p[:,i] = np.sum(pST[:,i,1:], axis = 1) / den[i]

    return p    
            
# A. Forward algorithm
@jit
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
    a = np.zeros((S, T))

    for s in range(S):
        a[s,:] = f[s, 0] / S

    aS = np.repeat(np.sum(a[:,0]), T)
    aR = np.repeat(a[:,0] / aS[0], T).reshape(S, T)

    """
    Fill out for t in [1, T]
    """
    for t in range(1, T):
        """
        (1) p_ij: transition from i to j
        (2) p[j,i] each column is where we come from, each row where we go to
        (3) Hence: p_ij = p[j,i]
        """
        for j in range(S):
            a[j,t] = f[j,t] * np.sum(p[j,:] * aR[:, t-1])

        aS[t]    = np.sum(a[:, t])
        aR[:, t] = a[:,t] / aS[t]
        
    return aR, aS

# B. Backward algorithm
@jit
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

    Also:
        (1) p_ij: transition from i to j
        (2) p[j,i] each column is where we come from, each row where we go to
        (3) Hence: p_ij = p[j,i]
    """
    for t in range(T - 2, -1, -1):
        for i in range(S):
            b[i, t] = np.sum(bR[:,t+1] * f[:, t+1] * p[:, i])
        bR[:, t] = b[:, t] / np.sum(b[:,t])

    return np.array(bR)

# Output (smoothed) p1, p2, ..., pN
@jit(nopython = True)
def pStarFct(T, S, aR, bR):
    pStar = np.zeros((S, T))
    for t in range(T):
        den   = np.sum(bR[:, t] * aR[:, t]) # den = denominator
        pStar[:,t] = bR[:, t] * aR[:,t] / den
    return pStar

# Output (smoothed transition) p11, p12, ..., p1N, p21, p22, ..., p2N, pN1, pN2, ..., pNN
@jit(nopython = True)
def pStarTFct(f, T, S, aR, aS, bR, p):
    pStarT = np.zeros((S,S,T))
    pStarT[:,:,0] = p / S
    for t in range(1, T):
        den = aS[t] * np.sum(bR[:,t] * aR[:,t])
        for i in range(S):
            pStarT[i, :,t] = bR[:,t] * f[:,t] * p[:,i] * aR[i, t-1] / den
    
    return pStarT

# E. Expected log-likelihood function to maximise
@jit(nopython = True)
def logLikFct(y, f, p, pS, pST, S):
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    
    # first sum (V.13), page 13
    sum_pIJ = 0
    for i in range(S):
        for j in range(S):
            sum_pST = np.sum(pST[j, i, 1:])
            sum_pIJ += np.log(p[j, i]) * sum_pST

    # Second sum (V.13), page 13
    sum_pJ = 0
    for j in range(S):
        sum_pJ += np.sum(pS[j, :] * np.log(f[j, :]))

    return k + sum_pIJ + sum_pJ

# ============================================= #
# ===== EM Algorithm ========================== #
# ============================================= #

@jit
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

    aR, aS = aFct(mat, states, f, p)
    bR      = bFct(mat, states, f, p)

    pStar    = pStarFct(mat, states, aR, bR)
    pStarT   = pStarTFct(f, mat, states, aR, aS, bR, p)

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