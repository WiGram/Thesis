"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.
"""

import EM_NM_EX as em
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
np.set_printoptions(suppress=True)   # Disable scientific notation

# ============================================= #
# ===== Parameter estimation ))))============== #
# ============================================= #

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis=1)
colNames = excessMRets.columns
A = len(colNames)  # Assets
y = np.array(excessMRets.T)  # Returns

trials = 200
S = 3  # States
T = len(y[0, :])  # Periods
p = np.repeat(1.0 / S, S * S).reshape(S, S)
pS = np.random.uniform(size=S * T).reshape(S, T)

#  Multivariate
ms, vs, ps, llh, pStar, pStarT = em.multEM(y, trials, T, S, A, p, pS)

#  Retrieve parameter estimates
probs = ps[trials-1]  # Transition probabilities
cov = vs[trials-1]  # One for each state, size: (States x (Asset x Asset))
mu = ms[trials-1]  # Size (Asset x states)
rf = rf[T-1]     # Risk-free rate in the last period

# ============================================= #
# ===== Implementation ======================== #
# ============================================= #

start = 1    # Start state, 1: Crash, 2: Slow Growth, 3: Bull
T = 1        # Investment horizon
G = 5        # Risk aversion (gamma)
M = 1000     # Simulated state paths
N = 1        # Simulated return processes
W = 10000    # [w]eight[S]ims
ApB = A + 1  # Assets plus bank account
Tmax = 120   # Used to compute u for any length

# Read documentation; the following q exits documentation
# help(owm.findOptimalWeights)
np.random.seed(12345)
u = np.random.uniform(0,1,(M,Tmax))
Ret, states = ssr.returnSim(S, M, N, A, start, mu, cov, probs, Tmax, u)

R = list()
maturities = [1,2,3,6,9,12,18,24,36,48,60,72,84,96,108,120]

for mat in maturities:
    R.append(Ret[:,:,:mat])

weights = np.random.random(size = (ApB, W))
wM = pwm.pfWeights(weights)

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