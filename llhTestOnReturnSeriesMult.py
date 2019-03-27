"""
Date:    March 10th, 2019
Authors: Kristian Strand and William Gram
Subject: MLE of multivariate models.

Description:
We find likelihood value by maximum likelihood estimation
on multivariate models.
"""

import genData as gd
import EM_NM_EX as em
import emPlots as emp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
np.set_printoptions(suppress = True)   # Disable scientific notation

""" Initial functions """
def density(d, dR, covm, A):
        return 1.0 / np.sqrt( (2.0 * np.pi) ** A * d) * np.exp(-0.5 * dR.T.dot(np.linalg.inv(covm)).dot(dR))

def llhFct(params, y):
    A = y.shape[0] # Assets
    T = y.shape[1]
    """
    Påvirker også VAR-matricen.... Husk dette!
    if VAR == False:
        T = y.shape[0] # Time periods
    else:
        T = y.shape[0] - 1
    """
    #
    mu   = params[0:A]
    #
    idx = np.tril_indices(A)     # indexes the triangular matrix elements
    chol = np.zeros((A,A))       # Generates 0 matrix of dim AxA
    chol[idx] = params[A:]       # Fills the lower triangular matrix
    for i in range(A):
        chol[i,i] = np.exp(chol[i,i])  # Ensures positive diagonal
    covm = np.dot(chol, chol.T)  # Construct covariance matrix
    #
    """
    d:   determinant
    dR:  demeaned returns
    """
    #
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((A,T))
    for a in range(A):
        dR[a,:] = y[a,:] - mu[a]
    #
    llh = np.zeros(T)
    for t in range(T):
        llh[t] = density(d, dR[:,t], covm, A)
    #
    return -np.sum(np.log(llh))

def llhFctAR(params, y):
    A = y.shape[0]      # Assets
    T = y.shape[1] - 1  # Due to AR coefficients, the first period is NA
    #
    y_lead = y[:,1:]  # Technically this is until T + 1
    y_lag  = y[:,:T]  # T is already subtracted by 1
    #
    mu   = params[0:A]
    #
    ar   = params[A:2*A]
    #
    idx = np.tril_indices(A)     # indexes the triangular matrix elements
    chol = np.zeros((A,A))       # Generates 0 matrix of dim AxA
    chol[idx] = params[2*A:]       # Fills the lower triangular matrix
    for i in range(A):
        chol[i,i] = np.exp(chol[i,i])  # Ensures positive diagonal
    covm = np.dot(chol, chol.T)  # Construct covariance matrix
    #
    """
    d:   determinant
    dR:  demeaned returns
    """
    #
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((A,T))
    for a in range(A):
        dR[a,:] = y_lead[a,:] - mu[a] - ar[a] * y_lag[a,:]
    #
    llh = np.zeros(T)
    for t in range(T):
        llh[t] = density(d, dR[:,t], covm, A)
    #
    return -np.sum(np.log(llh))

def llhFctX(params, y, x):
    A = y.shape[0]  # Assets
    T = y.shape[1]
    #
    mu   = params[0:A]
    #
    beta = params[A:2*A]
    #
    idx = np.tril_indices(A)     # indexes the triangular matrix elements
    chol = np.zeros((A,A))       # Generates 0 matrix of dim AxA
    chol[idx] = params[2*A:]       # Fills the lower triangular matrix
    for i in range(A):
        chol[i,i] = np.exp(chol[i,i])  # Ensures positive diagonal
    covm = np.dot(chol, chol.T)  # Construct covariance matrix
    #
    """
    d:   determinant
    dR:  demeaned returns
    """
    #
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((A,T))
    for a in range(A):
        dR[a,:] = y[a,:] - mu[a] - beta[a] * x
    #
    llh = np.zeros(T)
    for t in range(T):
        llh[t] = density(d, dR[:,t], covm, A)
    #
    return -np.sum(np.log(llh))

def llhFctXAR(params, y, x):
    A = y.shape[0]  # Assets
    T = y.shape[1] - 1
    #
    y_lead = y[:,1:]  # Technically this is until T + 1
    y_lag  = y[:,:T]  # T is already subtracted by 1
    x      = x[1:]    # Make exogenous same dimension as y
    #
    mu   = params[0:A]
    #
    ar   = params[A:2*A]
    #
    beta = params[2*A:3*A]
    #
    idx = np.tril_indices(A)     # indexes the triangular matrix elements
    chol = np.zeros((A,A))       # Generates 0 matrix of dim AxA
    chol[idx] = params[3*A:]       # Fills the lower triangular matrix
    for i in range(A):
        chol[i,i] = np.exp(chol[i,i])  # Ensures positive diagonal
    covm = np.dot(chol, chol.T)  # Construct covariance matrix
    #
    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((A,T))
    for a in range(A):
        dR[a,:] = y_lead[a,:] - mu[a] - ar[a] * y_lag[a,:] - beta[a] * x
    #
    llh = np.zeros(T)
    for t in range(T):
        llh[t] = density(d, dR[:,t], covm, A)
    #
    return -np.sum(np.log(llh))

""" End of functions """

# ============================================= #
# ===== Generate data ========================= #
# ============================================= #

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
A = len(colNames) # Assets
y = np.array(excessMRets.T) # Returns

sims = 200
S    = 3 # States
T    = len(y[0,:]) # Periods
p    = np.repeat(1.0 / S, S * S).reshape(S, S)
pS   = np.random.uniform(size = S * T).reshape(S, T)

# Multivariate
ms, vs, ps, llh, pStar, pStarT = em.multEM(y, sims, T, S, A, p, pS)

# ============================================= #


# ============================================= #
# ===== Estimate linear multivariate model ==== #
# ============================================= #

# params = [mu0, mu1, ..., muA, covm[1,1], covm[1,2], ..., cov[1,N], cov[2,1], ... , cov[2,N], ..... ]

# ============================================= #
# 1. Normal linear model
mu  = ms[sims-1][:,2]   # Initial mean parameter guess
cov = vs[sims-1,2,:,:]  # Initial covariance guess
chol = np.linalg.cholesky(cov)  # Cholesky factorise
for i in range(A):
    chol[i,i] = np.log(chol[i,i])  # Algorithm takes exponential of diagonal

idx = np.tril_indices(A)  # Takes indices of lower triangular elements
cholPars = chol[idx]      # Squashes lower triangular matrix into flat vector
params = np.concatenate((mu, cholPars))  # Combines model parameters

# Test for sensible likelihood value
llhFct(params, y)

# Maximise negative value = minimise positive value
resNormal = minimize(llhFct, params, y, method = 'L-BFGS-B')

# Return optimised likelihood value
resNormal.fun

# Retrieve parameter values
muEst = resNormal.x[:A]
cholEst = np.zeros((A,A))
cholEst[idx] = resNormal.x[A:]
for i in range(A):
    cholEst[i,i] = np.exp(cholEst[i,i])

covmEst = np.dot(cholEst, cholEst.T)
muEst
covmEst

aic_normal  = 2 * len(params) + 2 * resNormal.fun  # minus negative of likelihood
bic_normal  = np.log(y.shape[1])*len(params) + 2 * resNormal.fun
hqic_normal = 2 * resNormal.fun + 2 * len(params) * np.log(np.log(y.shape[1]))
# ============================================= #

# ============================================= #
# 2. AR(1)-model, no lagged cross-dependancy

mu  = ms[sims-1][:,2]   # Initial mean parameter guess
ar  = np.random.uniform(low = 0.1, high = 0.3, size = A)
chol = np.linalg.cholesky(cov)  # Cholesky factorise
for i in range(A):
    chol[i,i] = np.log(chol[i,i])  # Algorithm takes exponential of diagonal

idx = np.tril_indices(A)  # Takes indices of lower triangular elements
cholPars = chol[idx]      # Squashes lower triangular matrix into flat vector
params = np.concatenate((mu, ar, cholPars))  # Combines model parameters

# Test for sensible likelihood value
llhFctAR(params, y)

# Maximise negative value = minimise positive value
resAR = minimize(llhFctAR, params, y, method = 'L-BFGS-B')

# Return optimised likelihood value
resAR.fun

# Retrieve parameter values
muEst = resAR.x[:A]
arEst = resAR.x[A:2*A]
cholEst = np.zeros((A,A))
cholEst[idx] = resAR.x[2*A:]
for i in range(A):
    cholEst[i,i] = np.exp(cholEst[i,i])

covmEst = np.dot(cholEst, cholEst.T)
muEst
arEst
covmEst

aic_ar  = 2 * len(params) + 2 * resAR.fun  # minus negative of likelihood
bic_ar  = np.log(y.shape[1])*len(params) + 2 * resAR.fun
hqic_ar = 2 * resAR.fun + 2 * len(params) * np.log(np.log(y.shape[1]))
# ============================================= #


# ============================================= #
# 3. Exogenous model

mu  = ms[sims-1][:,2]   # Initial mean parameter guess
ex  = np.random.uniform(low = 0.1, high = 0.3, size = A)
chol = np.linalg.cholesky(cov)  # Cholesky factorise
for i in range(A):
    chol[i,i] = np.log(chol[i,i])  # Algorithm takes exponential of diagonal

idx = np.tril_indices(A)  # Takes indices of lower triangular elements
cholPars = chol[idx]      # Squashes lower triangular matrix into flat vector
params = np.concatenate((mu, ex, cholPars))  # Combines model parameters

# Read exogenous variable (Dividend Yield)
x  = pd.read_excel('/home/william/Dropbox/KU/K4/Python/DivYield.xlsx', 'Monthly')
x  = np.array(x.iloc[:,1])

# Combine assets with exogenous
args = y, x

# Test for sensible likelihood value
llhFctX(params, *args)

# Maximise negative value = minimise positive value
resX = minimize(llhFctX, params, args = args, method = 'L-BFGS-B')

# Return optimised likelihood value
resX.fun

# Retrieve parameter values
muEst = resX.x[:A]
exEst = resX.x[A:2*A]
cholEst = np.zeros((A,A))
cholEst[idx] = resX.x[2*A:]
for i in range(A):
    cholEst[i,i] = np.exp(cholEst[i,i])

covmEst = np.dot(cholEst, cholEst.T)
muEst
exEst
covmEst

aic_ex  = 2 * len(params) + 2 * resX.fun  # minus negative of likelihood
bic_ex  = np.log(y.shape[1])*len(params) + 2 * resX.fun
hqic_ex = 2 * resX.fun + 2 * len(params) * np.log(np.log(y.shape[1]))
# ============================================= #


# ============================================= #
# 4. Exogenous model with AR(1) diagonals only

mu  = ms[sims-1][:,2]   # Initial mean parameter guess
ar  = np.random.uniform(low = 0.1, high = 0.3, size = A)
ex  = np.random.uniform(low = 0.1, high = 0.3, size = A)
chol = np.linalg.cholesky(cov)  # Cholesky factorise
for i in range(A):
    chol[i,i] = np.log(chol[i,i])  # Algorithm takes exponential of diagonal

idx = np.tril_indices(A)  # Takes indices of lower triangular elements
cholPars = chol[idx]      # Squashes lower triangular matrix into flat vector
params = np.concatenate((mu, ar, ex, cholPars))  # Combines model parameters

# Read exogenous variable (Dividend Yield)
# x  = pd.read_excel('/home/william/Dropbox/KU/K4/Python/DivYield.xlsx', 'Monthly')
# x  = np.array(x.iloc[:,1])

# Combine assets with exogenous
args = y, x

# Test for sensible likelihood value
llhFctXAR(params, *args)

# Maximise negative value = minimise positive value
resXAR = minimize(llhFctXAR, params, args = args, method = 'L-BFGS-B')

# Return optimised likelihood value
resXAR.fun

# Retrieve parameter values
muEst = resXAR.x[:A]
arEst = resXAR.x[A:2*A]
exEst = resXAR.x[2*A:3*A]
cholEst = np.zeros((A,A))
cholEst[idx] = resXAR.x[3*A:]
for i in range(A):
    cholEst[i,i] = np.exp(cholEst[i,i])

covmEst = np.dot(cholEst, cholEst.T)
muEst
arEst
exEst
covmEst

aic_ex_ar  = 2 * len(params) + 2 * resXAR.fun  # minus negative of likelihood
bic_ex_ar  = np.log(y.shape[1])*len(params) + 2 * resXAR.fun
hqic_ex_ar = 2 * resXAR.fun + 2 * len(params) * np.log(np.log(y.shape[1]))
# ============================================= #

d = {'Normal':      [resNormal.fun,aic_normal,bic_normal,hqic_normal], 
     'AR(1)':       [resAR.fun,aic_ar,bic_ar,hqic_ar],
     'Exog.':       [resX.fun,aic_ex,bic_ex,hqic_ex],
     'AR(1), Exog.':[resXAR.fun,aic_ex_ar,bic_ex_ar,hqic_ex_ar]}
df = pd.DataFrame(data = d,index = ['Likelihood Value','AIC','BIC','HQIC'])

# df.to_latex()
