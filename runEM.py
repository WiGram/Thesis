"""
Date:    February 11th, 2019
Authors: Kristian Strand and William Gram
Subject: Describing Bloomberg time series

Description:
This script runs the EM algorithm by calling the EM_NM_EX.py script
by generating data in the genData.py script. Finally the output is
plotted by calling the plotsEM.py script.
"""

import genData as gd
import EM_NM_EX as em
import emPlots as emp
import numpy as np
import pandas as pd
%matplotlib

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

"""
# ===== Monthly Absolute ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
monthlyRets = monthlyRets.drop(['S&P 500'], axis = 1)
colNames =.columns
assets = len(colNames)
y = np.array(monthlyRets.T)
"""

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

# Univariate
m, v, pp, l, pss, pst = em.uniEM(y[0], sims, T, S, p, pS)

# Plot all
emp.emPlots(sims, S, A, rDates, colNames, llh, ps, vs, ms, pStar, plots = 'likelihood')
emp.emPlots(sims, S, A, rDates, colNames, llh, ps, vs, ms, pStar, plots = 'all')
emp.emUniPlots(sims, S, rDates, colNames, l, pp, v, m, pss)



# Testing standard errors on likelihood function
# So far without success:

from llhFct import llhFct, llhUniFct, llhUniSumFct

# Multivariate
params = ms[sims-1], vs[sims-1], ps[sims-1][:S-1, :]
args   = y, pStar, pStarT
llhFct(params, *args)


params = np.concatenate((m[sims-1], v[sims-1], np.concatenate(pp[sims-1][:S-1,:])))
args = y[0], pss, pst

llhUniSumFct(params, *args)

import numdifftools as nd

hfct = nd.Hessian(llhUniSumFct)
h = hfct(params, *args)

jacf = nd.Jacobian(llhUniFct)
j = jacf(params, *args)


"""
Example of how to use numdifftools

def testFct(x, a, b):
    return - x ** 2 * a + b

testFct(3, 1, np.array([1,2,3,4]))

# Expected Output
# - 3 ** 2 * 1 + (1,2,3,4)
# - 9 + (1,2,3,4)
# = - (8,7,6,5)
#
# Actual output
# = - (8,7,6,5) => OKAY!

jacf = nd.Gradient(testFct)
jacf(2, np.array([2,3,4]), 1)

# EXPECTED OUTPUT
# f' of (- x ** 2) * [2,3,4]
# = -2 * x * [2,3,4]
# = -2 * 2 * [2,3,4]
# = -8, - 12, -16

# OUTPUT: -8, - 12, -16 => OKAY!

def testTwoFct(vars, args):
    a = args[0]
    b = args[1]
    xes = args[2]
    x = vars[:xes]
    y = vars[xes:]

    return - x ** 2 * a + y ** 2 *b

testTwoFct(2,3,1,1)
# Output: 5

x = np.array([2,3])
y = np.array([2])

vars = np.concatenate([x,y])

a = np.array([2])
b = np.array([2])
xes = np.array([len(x)])

args = np.concatenate([a,b, xes])

jacf = nd.Gradient(testTwoFct)
jacf(vars, args)

"""