""" 
Date:    March 27th, 2019
Authors: Kristian Strand and William Gram
Subject: Numerical optimisation on weights in a univariate model.

Description:
-----------------------------
We intend to find portfolio weights from a CRRA quadratic
utility function for a univariate model.

Current issues:
-----------------------------
1. This setup only works with 1 asset
2. Solutions are sensitive to sufficient return simulations

Imports:
-----------------------------
numpy as np
matplotlib.pyplot as plt
simulateSimsReturns as ssr
scipy.optimize as opt
numba.jit

Input parameters used (defaults in parentheses if available):
-----------------------------
S       (3)     amount of states
M       (50000) amount of simulated state paths
N       (1)     amount of simulated return paths per simulated state path
A       (1)     amount of assets beyond the risk free asset
T       (120)   amount of periods 
start   (1)     initial state to simulate state paths from
rf      (0.3)   risk free rate in percent (0.3 = 0.3%)
g       (5)     gamma - degree of risk aversion
seed    (12345) functions have default seed = 12345. Set this to change seed
mu      (S,1)   vector of mean returns in each state
cov     (S,1)   vector of variances in each state
probs   (S,S)   matrix of transition probabilities (transition: rows to cols)
u       (M*N,T) random uniform numbers for each period for all simulated paths
w       (0.6)   scalar of weight on risky asset (between 0. and 1.)
"""

import simulateSimsReturns as ssr
import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from scipy import optimize as opt

@jit(nopython = True)
def expectedUtility(w, returns, rf, g, T):
    riskfreeCompoundedReturn = np.exp(T * rf / 100)
    riskyCompoundReturn = np.exp(np.sum(returns/100, axis = 1))*riskfreeCompoundedReturn
    
    numerator = ((1 - w) * riskfreeCompoundedReturn + w * riskyCompoundReturn ) ** (1 - g)
    denominator = 1 - g
    
    return -np.mean(numerator / denominator)

S = 3
M = 150000
N = 1
A = 1
T = 120
start = 3
rf = 0.3
g = 5

# High Yield 3 state
mu  = np.array([0.72, 0.012, -2.89])
cov = np.array([1.07, 6.08,  32.24]) 
probs = np.array([
    [0.94,0.09,0.00],
    [0.06,0.88,0.15],
    [0.00,0.03,0.85]
])
"""
# Russell 1000 2 states
mu = np.array([1.1135,-1.1334])
cov = np.array([8.511359, 39.931593])
probs = np.array([
    [0.95,0.12],
    [0.05,0.88]
])
"""

# ============================================= #
# ===== Simulate return paths ================= #
# ============================================= #

u = np.random.uniform(0, 1, size=(M*N, T))
returns, states = ssr.returnSim(S,M,N,A,start,mu,cov,probs,T,u)

"""
Graphical analysis of returns
---------------
plt.plot(returns[0]); plt.plot(states[0]); plt.show()
"""

# ============================================= #
# ===== Initial optimisation ================== #
# ============================================= #

method = 'Nelder-Mead'
w = 0.06
args = returns, rf, g, T
results = opt.minimize(expectedUtility, w, args = args, method = method)
expectedUtil = results.fun
optimisingW = results.x
print(-expectedUtil, optimisingW)

# ============================================= #
# ===== Optimisation for various g's and T's == #
# ============================================= #

gamma = np.array([3,5,7,9,12])
maturities = np.array([1,2,3,6,9,12,15,18,21,24,30,36,42,48,54,60,72,84,96,108,120])
weights = np.repeat(0.6,len(maturities))

R = [returns[:,:mat] for mat in maturities]

for g in gamma:
    for i in range(len(maturities)):
        args = R[i], rf, g, maturities[i]
        weights[i] = opt.minimize(expectedUtility, w, args = args, method = method).x
    plt.plot(maturities[5:], weights[5:])
plt.show()


"""
# Alternative approach: simulate many weights
W  = 1000
wt = np.random.random(W)

w = 0.06
T = 96
args = returns[:,:T-1], rf, g, T
results = opt.minimize(expectedUtility, w, args = args)
expectedUtil = results.fun
optimisingW = results.x
print(-expectedUtil, optimisingW)
expectedUtility(0.06, *args)

for i in range(len(maturities)):
    args = R[i], rf, g, maturities[i]
    weights[i] = opt.minimize(expectedUtility, w, args = args).x

plt.plot(maturities[5:], weights[5:])
plt.show()
"""