"""
Date:    February 28th, 2019
Authors: Kristian Strand and William Gram
Subject: Parameter estimation of pseudo-residuals

Description:
ML estimation of parameters for different model
specifications of pseudo-residuals. The purpose is
to test whether the models generating the pseudo-
residuals is well-specified, which is tested through
tests on the properties of the pseudo-residuals.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# ============================================= #
# ===== Model functions ======================= #
# ============================================= #

def testStat(lStd, lTest):
    return - 2 * (lStd - lTest)

def pValue(testStat, type = 'normal'):
    if type == 'normal':
        return ValueError('WIP')

def densityFct(z, mean, vol):
    return 1 / np.sqrt(2 * np.pi * vol ** 2) * np.exp(-0.5 * (z - mean)**2 / vol ** 2)

def modelStd(z):
    return - np.sum(  np.log(densityFct(z, 0, 1)))

def modelNorm(params, z):
    alpha = params[0]  # alpha
    gamma  = params[1]  # sigma
    #
    mean = alpha
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z, mean, vol))  )

def modelARone(params, z):
    z_lead = z[1:len(z) - 0]
    z_lag  = z[0:len(z) - 1]
    #
    alpha = params[0]
    arOne = params[1]
    gamma = params[2] # sigma = exp(gamma) <= ensures sigma > 0
    # 
    mean = alpha + arOne * z_lag
    vol  = np.exp(gamma)
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwo(params, z):
    z_lead = z[2:len(z) - 0]
    z_lag  = z[1:len(z) - 1]
    z_llag = z[0:len(z) - 2]
    # 
    alpha = params[0]
    arOne = params[1]
    arTwo = params[2]
    gamma = params[3]
    #
    mean = alpha + arOne*z_lag + arTwo*z_llag
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwoS(params, z):
    z_lead = z[2:len(z) - 0]
    z_lag  = z[1:len(z) - 1]
    z_llag = z[0:len(z) - 2]
    # 
    alpha = params[0]
    arOne = params[1]
    arTwo = params[2]
    arOneS = params[3]
    arTwoS = params[4]
    gamma = params[5]
    #
    mean = alpha + arOne*z_lag + arTwo*z_llag + arOneS*(z_lag**2) + arTwoS*(z_llag**2)
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log( densityFct(z_lead, mean, vol) )  )

def modelX(params, z, ex):
    alpha = params[0]
    beta  = params[1]
    sigma = params[2]
    #
    mean = alpha + beta * ex
    vol  = sigma
    #
    return - np.sum(  np.log(densityFct(z, mean, vol))   )


# ============================================= #
# ===== Tests ================================= #
# ============================================= #

"""
# Initial parameters
length = 10000
eps = np.random.normal(size = length)
ex  = np.random.normal(size = length) # sim of some exogenous param

# PATH SIMULATIONS:

# 1. NORMAL:
x_par = np.array([0.8, 1.2])
x   = 0.8 + 1.2 * eps

# 2. ARone:
y_par = np.array([0.8, 0.5, 1.2])
y = np.ones(length)
for t in range(length-1):
    y[t+1] = 0.8 + 0.5 * y[t] + 1.2 * eps[t+1]

# 3. ARtwoS:
r_par = np.array([0.8, 0.5, -0.5, -0.05, -0.05, 1.2])
r = np.ones(length)
for t in range(length-2):
    r[t+2] = 0.8 + 0.5 * r[t+1] - 0.5 * r[t] - 0.05 * r[t+1] ** 2 -0.05 * r[t] ** 2 + 1.2 * eps[t+2]

# 4. X:
e_par = np.array([0.8, 0.5, 1.2])
e = 0.8 + 0.5 * ex + 1.2 * eps

# ============================================= #
# ===== Graphical test on simulated paths ===== #
# ============================================= #

# Are they all on the same scale and exhibit good behaviour?
plt.plot(x, alpha = 0.8)
plt.plot(y, alpha = 0.6)
plt.plot(r, alpha = 0.4)
plt.plot(e, alpha = 0.2)
plt.show()

# ============================================= #
# ===== Tests on parameters and likelihood ==== #
# ============================================= #

#  Normal
modelNorm(parNorm, x)
res_test = minimize(modelNorm, parNorm, args = x, method = 'L-BFGS-B')
np.round(res_test.x, 2), x_par
res_test.fun, modelNorm(res_test.x, x)

#  AR one
res_test = minimize(modelARone, parARone, args = y, method = 'L-BFGS-B')
np.round(res_test.x, 2), y_par
res_test.fun, modelARone(res_test.x, y)

#  AR two with quadratid terms
res_test = minimize(modelARtwoS, parARtwoS, args = r, method = 'L-BFGS-B')
np.round(res_test.x, 2), r_par
res_test.fun, modelARtwo(res_test.x, r)

#  X
args = e, ex
res_test = minimize(modelX, parX, args = args, method = 'L-BFGS-B')
np.round(res_test.x, 2), e_par
res_test.fun, modelX(res_test.x, *args) # "*" 'unpacks' (args = e, ex)
"""

# ============================================= #
# ===== Data load ============================= #
# ============================================= #

"""
# 2-State (had different excel set-up, hence treated diff)
# EDIT PATH!!!
path = '/home/william/Dropbox/KU/K4/Python/HY2.xlsx'
xls  = pd.ExcelFile(path)
z11  = pd.read_excel(xls, 'Transformed z-score HMM_11(2)')
z01  = pd.read_excel(xls, 'Transformed z-score HMM_01(2)')
z10  = pd.read_excel(xls, 'Transformed z-score HMM_10(2)')
z11x = pd.read_excel(xls, 'Transformed z-score HMMX_11(2)')

# Generate one joint table to iterate through
zs = pd.concat([z11,z01,z10,z11x], axis=1, join_axes=[z11.index])
"""

path = '/home/william/Dropbox/KU/K4/Python/HY3.xlsx'
xls  = pd.ExcelFile(path)
zs    = pd.read_excel(xls, 'Sheet5')
z11  = zs.iloc[:,1]
z01  = zs.iloc[:,1]
z10  = zs.iloc[:,1]
z11x = zs.iloc[:,1]

# test: index 0 should be a number, and column name should be 'z'
z11[:2]
z01[:2]
z10[:2]
z11x[:2]
# ============================================= #

# ============================================= #
# ===== Data inspection ======================= #
# ============================================= #

testTitles = np.array(['Z(1,1)','Z(0,1)','Z(1,0)','Z(1,1)(X)'])
fig, axes = plt.subplots(nrows = 2,ncols = 2,
                         sharex = True,
                         sharey = True,
                         figsize = (15,6))
pseudoRes = np.array([zs.iloc[:,j] for j in range(zs.shape[1])])
for ax, title, y in zip(axes.flat, testTitles, pseudoRes):
    ax.plot(range(zs.shape[0]), y)
    ax.set_title(title)
    ax.grid(False)

plt.show()

# Sanity check - three are basically identical: Turns out they just basically are.
plt.plot(zs.iloc[:,0])
plt.plot(zs.iloc[:,1])
plt.plot(zs.iloc[:,3])
plt.show()

# Index 2 (Z10) behaves very differently in variance!! Treat separately
idx = np.array([0,1,3])

# ============================================= #


# ============================================= #
# ===== Likelihood retributions =============== #
# ============================================= #

# Initial parameters: Values are guided by sporadic minimisation tests
parNorm   = np.array([-0.04, 1.0])
parARone  = np.array([0.04, 0.6, 1.0])
parARtwoS = np.array([0.05, 0.4, 0.4, -0.1, -0.1, 1.0])
parX      = np.array([0.5, 0.4, 0.8])

# Contain lists containing parameters and likelihood values, resp.
parsN = np.zeros((4,2))
parsO = np.zeros((4,3))
parsT = np.zeros((4,6))
parsX = np.zeros((4,3))
llhS = np.zeros(4)
llhN = np.zeros(4)
llhO = np.zeros(4)
llhT = np.zeros(4)
llhX = np.zeros(4)
tN = np.zeros(4)
tO = np.zeros(4)
tT = np.zeros(4)
tX = np.zeros(4)

method = 'L-BFGS-B'
for i in range(zs.shape[1]):
    # Standard computation
    llhS[i] = - modelStd(zs.iloc[i])
    # Normal test
    res = minimize(modelNorm, parNorm, args = zs.iloc[:,i], method = method)
    parsN[i] = np.hstack(   ( res.x[:1], np.exp(res.x[1]))  )
    llhN[i] = -res.fun
    tN[i] = testStat(llhS[i], llhN[i])
    # AR(1) test
    res = minimize(modelARone, parARone, args = zs.iloc[:,i], method = method)
    parsO[i] = np.hstack(   ( res.x[:2], np.exp(res.x[2]) )   )
    llhO[i] = -res.fun
    tO[i] = testStat(llhS[i], llhO[i])
    # AR(2) test with quadratic second lag
    res = minimize(modelARtwoS, parARtwoS, args = zs.iloc[:,i], method = method)
    parsT[i] = np.hstack(   ( res.x[:5], np.exp(res.x[5]) )   )
    llhT[i] = -res.fun
    tT[i] = testStat(llhS[i], llhT[i])

print(llhS)
print(llhN)
print(llhO)
print(llhT)

print(parsN)
print(parsO)
print(parsT)

print(tN)
print(tO)
print(tT)
