""" 
Date:    March 27th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights using numerical optimisation.

Description:
-----------------------------
We intend to find portfolio weights from a CRRA quadratic
utility function.

Current issues:
-----------------------------
1. Not all start-states provide meaningful weights.
2. The setup has not been tested beyond 2 assets.

Imports:
-----------------------------
numpy as np
matplotlib.pyplot as plt
simulateSimsReturns as ssr
constrainedOptimisation as copt
scipy.optimize as opt
numba.jit

Input parameters used (defaults in parentheses if available):
-----------------------------
S       (2)     amount of states
M       (50000) amount of simulated state paths
N       (1)     amount of simulated return paths per simulated state path
A       (2)     amount of assets beyond the risk free asset
T       (120)   amount of periods 
start   (1)     initial state to simulate state paths from
rf      (0.3)   risk free rate in percent (0.3 = 0.3%)
g       (5)     gamma - degree of risk aversion
seed    (12345) functions have default seed = 12345. Set this to change seed
mu      (A,S)   matrix of multivariate mean returns for each state
cov     (S,A,A) S matrices of A x A covariance matrices
probs   (S,S)   matrix of transition probabilities (transition: rows to cols)
u       (M*N,T) random uniform numbers for each period for all simulated paths
w       (A,1)   vector of weights that must sum to weakly less than 1.0
"""

import simulateSimsReturns as ssr
import constrainedOptimiser as copt
import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from scipy import optimize as opt

@jit(nopython = True)
def expectedUtilityMult(w, returns, rf, g, A, T):
    """
    Description
    -----------------------------
    Computes expected utility of wealth.
    Wealth is compounded return on risky and risk free assets.
    Utility is (W)^(1-gamma) / (1-gamma) -> gamma != 1 !!
    
    Arguments
    -----------------------------
    w           (A+1,1) vector of standardised weights, default is (.4,.25,.35)
    returns     (M*N,A,T) multivariate simulations, in percent, not decimals
    rf          Scalar in percent, not decimals, default is 0.3 (percent)
    g           Scalar indicating gamma, risk aversion, default is 5
    A           Scalar indicating amount of assets excl. bank account
    T           Periods in months, default is 120
    """
    
    # rfCR: risk free compounded return
    # rCR:  risky compounded return
    
    rfCR = np.exp(T * rf / 100)
    denominator = 1 - g
    rCR = np.exp(np.sum(returns/100, axis=2))*rfCR
    numerator = (w[A] * rfCR + np.sum(w[:A] * rCR, axis=1)) ** (1 - g)
    return -np.mean(numerator / denominator) * 100000

@jit(nopython = True)
def quickEUM(w,returns,rf,g,A,T):
    """
    Description
    -----------------------------
    Computes expected utility of wealth for multiple sets of weights.
    Wealth is compounded return on risky and risk free assets.
    Utility is (W)^(1-gamma) / (1-gamma) -> gamma != 1 !!
    
    Arguments
    -----------------------------
    w           (A+1,1) vector of standardised weights, default is (.4,.25,.35)
    returns     (M*N,A,T) multivariate simulations, in percent, not decimals
    rf          Scalar in percent, not decimals, default is 0.3 (percent)
    g           Scalar indicating gamma, risk aversion, default is 5
    A           Scalar indicating amount of assets excl. bank account
    T           Periods in months, default is 120
    """
    W = len(w[0,:])
    expUtil = np.zeros(W)
    for i in np.arange(W):
        expUtil[i] = - expectedUtilityMult(w[:,i],returns,rf,g,A,T) / 100000
    return expUtil

S     = 2
M     = 50000
N     = 1
A     = 2
ApB   = A + 1
T     = 120
start = 2
rf    = 0.6
g     = 5
seed  = 23456
muHY  = np.array([0.5902,-0.0451])
muR1  = np.array([1.1198,-1.1242])
mu    = np.array((muHY,muR1))
cov   = np.array([
    [[ 1.44590, 1.97827],
     [ 1.97827, 8.52906]],
    [[15.62243,15.34531],
     [15.34531,39.53987]]
])
probs = np.array([
    [0.93,0.16],
    [0.07,0.84]
])

# ============================================= #
# ===== Simulate state and return paths ======= #
# ============================================= #

u = np.random.uniform(0, 1, size=(M*N, T))
returns, states = ssr.returnSim(S,M,N,A,start,mu,cov,probs,T,u,seed=seed)

# ============================================= #
# ===== Optimisation for several g's and T's == #
# ============================================= #

w          = np.array([0.4,0.35,0.25])
gamma      = np.array([3,5,7,9,12])
maturities = np.array([1,2,3,6,9,12,15,18,21,24,30,36,42,48,54,60,72,84,96,108,120])
weights    = np.array(list(zip(
    np.repeat(w[0],len(maturities)),
    np.repeat(w[1],len(maturities)),
    np.repeat(w[2],len(maturities))
)))

R = [returns[:,:,:mat] for mat in maturities]

#for g in gamma:
for i in range(len(maturities)):
    args = R[i], rf, g, A, maturities[i]
    results = copt.constrainedOptimiser(expectedUtilityMult,w,args,ApB)
    weights[i] = results.x
plt.plot(maturities, weights[:,0], label = 'hy')
plt.plot(maturities, weights[:,1], label = 'r1')
plt.plot(maturities, weights[:,2], label = 'rf')
plt.legend()
plt.show()

# ============================================= #
# ===== Diverse checks of the data ============ #
# ============================================= #

"""
Graphical check (seemingly useful):
-----------------------------
import pandas as pd
hy = pd.DataFrame(returns[:100,0,:].T)
r1 = pd.DataFrame(returns[:100,1,:].T)
st = pd.DataFrame(states[:100,:].T)

hy.plot(color='grey',legend=False,alpha=0.3)
plt.plot(range(T),hy.iloc[:,50],color='blue',linewidth=2,alpha=.7)
plt.show()

r1.plot(color='grey',legend=False,alpha=0.3)
plt.plot(range(T),r1.iloc[:,50],color='blue',linewidth=2,alpha=.7)
plt.show()

st.plot(color='grey',legend=False,alpha=0.3)
plt.plot(range(T),st.iloc[:,50],color='blue',linewidth=2,alpha=.7)
plt.show()
"""

"""
# ============================================= #
# ===== Experiment with T = 120, g = 5 ======== #
# ============================================= #

w    = np.array([0.4,0.25,0.35])
args = returns,rf,g,A,T

# Optimisation step
results = copt.constrainedOptimiser(expectedUtilityMult,w,args,ApB)

# Derive portfolio return (.fun) and optimal weights (.x)
expectedUtil = results.fun
optimisingW = results.x
print(-expectedUtil, optimisingW)
"""

"""
Graphical and grid search check:
-----------------------------
W = 1000
w = np.random.random(size = (ApB,W))
colSum = np.sum(w,axis = 0)
w /= colSum
testEU = quickEUM(w,returns,rf,g,A,T)

w[:,np.argmax(testEU)]

x = w[0,:]
y = w[1,:]
z = testEU

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,c='r',marker='o')
ax.set_xlabel('High Yield allocation')
ax.set_ylabel('Russell 1000 allocation')
ax.set_zlabel('Expected Utility')
plt.show()
"""

# ============================================= #
# ===== The end =============================== #
# ============================================= #