"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.

Imports:
EM_NM_EX as em
sharpe as sp
numpy as np
pandas as pd
quandl
matplotlib.pyplot as plt
numba.jit
pandas_datareader.data as web
genData as gd
simulateSimsReturns as ssr
pfWeightsModule as pwm
expUtilModule as eum
"""

# ============================================= #
# ===== Imports and global settings =========== #
# ============================================= #

import EM_NM_EX as em
import sharpe as sp
import numpy as np
import pandas as pd
import quandl
from matplotlib import pyplot as plt
from numba import jit
from pandas_datareader import data as web
import genData as gd
import simulateSimsReturns as ssr
import pfWeightsModule as pwm
import expUtilModule as eum
np.set_printoptions(suppress=True) # Disable scientific notation

# ============================================= #
# ===== Data collection ======================= #
# ============================================= #

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ===== Monthly excess returns ===== #
excessMRets = excessMRets.drop(['S&P 500'], axis=1)
colNames = excessMRets.columns
A = len(colNames)               # Assets
y = np.array(excessMRets.T)     # Returns

# ============================================= #
# ===== Model parameter estimation ============ #
# ============================================= #

# Defining model inputs
trials = 200
S = 3                           # States
T = len(y[0, :])                # Periods
p = np.repeat(1.0 / S, S * S).reshape(S, S)
pS = np.random.uniform(size=S * T).reshape(S, T)

# Run the model
ms, vs, ps, llh, pStar, pStarT = em.multEM(y, trials, T, S, A, p, pS)

#  Retrieve parameter estimates (last observations have presumably converged)
probs = ps[trials-1]  # (S,S)   matrix of transition probabilities
cov = vs[trials-1]    # (S,A*A) matrix of covariances for each state
mu = ms[trials-1]     # (A,S)   matrix of returns for each state
rf = rf[T-1]          # (1,1)   scalar of the risk-free rate in the last period

# ============================================= #
# ===== Implementation ======================== #
# ============================================= #

start = 1    # Start state, 1: Crash, 2: Slow Growth, 3: Bull for S = 3
T = 1        # Investment horizon
G = 5        # Risk aversion (gamma)
M = 1000     # Simulated state paths
N = 1        # Simulated return processes
W = 50000    # Amount of portfolio allocations
ApB = A + 1  # Assets plus bank account
Tmax = 120   # Used to compute u for any length

# Read documentation: print(ssr.returnSim.__doc__)
np.random.seed(12345)
u = np.random.uniform(0,1,(M,Tmax))
rets, states = ssr.returnSim(S, M, N, A, start, mu, cov, probs, Tmax, u)
rets /= 100.0

# First two moments
returnMatrix = np.mean(rets, axis = 2)
covMatrix = np.array([np.cov(rets[i]) for i in range(M*N)])

# Consider the portfolio next
weights = np.random.random(size = (A, W))
weightMatrix = pwm.pfWeights(weights)

# Convenience: acronym variables
rM, cM, wM = returnMatrix, covMatrix, weightMatrix

# Compute Sharpe Ratios; print(sp.pfSR.__doc__)
pfW, pfSR, index, maxim, pfR, pfS = sp.pfSR(rM,cM,wM,M,N,W,T)
wOpt, srOpt = sp.numOptSR(np.random.random(A),rM,cM,M,N,T)

# Test the returned SR's; print(sp.pfSRopt.__doc__)
-sp.pfSRopt(pfW, rM, cM, M, N, T), pfSR
-sp.pfSRopt(wOpt, rM, cM, M, N, T), srOpt

optedRet  = sp.pfRetUniWeight(rM,wOpt,M,N,T)
optedSdev = sp.pfCovUniWeight(cM,wOpt,M,N,T)

pfS[index],pfR[index]
optedSdev, optedRet

plt.scatter(pfS,pfR)
plt.scatter(x = pfS[index], y = pfR[index], alpha = 0.5, color = 'red')
plt.scatter(x = optedSdev, y = optedRet, alpha = 0.5, color = 'black')
plt.annotate('max', xy = (pfS[index], pfR[index]))
plt.annotate('num max', xy = (optedSdev, optedRet))
plt.show()

""" 
For one period: assume existing Sharpe Ratio.
For two+ periods: reset Sharpe Ratio to what's computed over simulated period.
"""
R = list()
maturities = [2,3,6,9,12,18,24,36,48,60,72,84,96,108,120]

for mat in maturities:
    R.append(rets[:,:,:mat])


eU, uMax, uArgMax, wMax = eum.expectedUtility(M,N,W,T,rf,R[0],wM,G,ApB)
wMax

# eU, uMax, uArgMax, wMax, R, states, wM = owm.findOptimalWeights(
#     M, N, W, T, S, A, rf, G, start, mu, cov, probs, u, w)

data = {'Maturity: {}'.format(T): [M, N, W, T, S, G, start]}
rows = [
    'Simulated state paths',
    'Simulated return sets',
    'Simulated portfolio weights',
    'Simulated time periods',
    'Amount of states',
    'Risk aversion (gamma)',
    'Start period'
]

df = pd.DataFrame(data, index=rows)
pfw = {'T = {}'.format(T): wMax}
rows = colNames.append(pd.Index(['Risk free asset']))
rows.name = 'Optimal PF weights'
dfw = pd.DataFrame(pfw, index=rows)

maturities = maturities[1:]
wMax = np.ones((len(maturities), ApB))

for i, mat in enumerate(maturities):
    print(i)
    eU, uMax, uArgMax, wMax[i] = eum.expectedUtility(
        M,N,W,mat,rf,R[i],wM,G,ApB
    )
    print(uArgMax)
    dfw['Maturity: {}'.format(mat)] = pd.Series(wMax[i], index=dfw.index)


x = [1] + maturities
xi = [i for i in range(len(x))]


for i in range(ApB):
    dfw.iloc[i, :].plot()

plt.legend()
plt.xticks(xi, x)
plt.xlabel('Maturities')
plt.ylabel('weight')
plt.show()
