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
import expUtilNum as eun
from scipy import optimize as opt
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
M = 100      # Simulated state paths
N = 100      # Simulated return processes
ApB = A + 1  # Assets plus bank account
Tmax = 120   # Used to compute u for any length

# Read documentation; the following q exits documentation
# help(owm.findOptimalWeights)
np.random.seed(12345)
u = np.random.uniform(0,1,(M,Tmax))
Ret, states = ssr.returnSim(S, M, N, A, start, mu, cov, probs, Tmax, u)

R = list()
maturities = [1,2,3,4,5,6,7,8,9,10,11,12,15,18,21,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120]

for mat in maturities:
    R.append(Ret[:,:,:mat])

# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1.0

# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})

# 0-1 bounds for each weight
bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))

# Initial Guess (equal distribution)
args = M,N,T,rf,R[0],G
x = np.ones((100,ApB))
for i in range(100):
    w = np.random.random(ApB)
    x[i] = opt.minimize(eun.expectedUtility, w, args, bounds = bounds, constraints = cons).x

for i in range(ApB):
    plt.plot(x[:,i])

plt.legend()
plt.show()

# eU, uMax, uArgMax, wMax, R, states, wM = owm.findOptimalWeights(
#     M, N, W, T, S, A, rf, G, start, mu, cov, probs, u, w)

data = {'Maturity: {}'.format(T): [M, N,T,S,G,start]}
rows = [
    'Simulated state paths',
    'Simulated return sets',
    'Simulated time periods',
    'Amount of states',
    'Risk aversion (gamma)',
    'Start period'
]

df = pd.DataFrame(data, index=rows)
pfw = {'T = {}'.format(T): x}
rows = colNames.append(pd.Index(['Risk free asset']))
rows.name = 'Optimal PF weights'
dfw = pd.DataFrame(pfw, index=rows)

df
dfw

maturities = maturities[1:]
weights = np.ones((len(maturities), ApB))

for i, mat in enumerate(maturities):
    args = M,N,T,rf,R[i+1],G
    weights[i] = opt.minimize(eun.expectedUtility, w, args, bounds = bounds, constraints = cons).x
    dfw['Maturity: {}'.format(mat)] = pd.Series(weights[i], index=dfw.index)


x = [1] + maturities
xi = [i for i in range(len(x))]


for i in range(ApB):
    dfw.iloc[i, :].plot()

plt.legend()
plt.xticks(xi, x)
plt.xlabel('Maturities')
plt.ylabel('weight')
plt.show()
