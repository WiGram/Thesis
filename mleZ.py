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

def densityFct(z, mean, vol):
    return 1 / np.sqrt(2 * np.pi * vol ** 2) * np.exp(-0.5 * (z - mean)**2 / vol ** 2)

def modelNorm(params, z):
    mean = params[0]  # alpha
    vol  = params[1]  # sigma

    return - np.sum(  np.log(densityFct(z, mean, vol))  )

def modelARone(params, z):
    z_lead = z[1:len(z) - 0]
    z_lag  = z[0:len(z) - 1]
    
    alpha = params[0]
    arOne = params[1]
    sigma = params[2]
    
    mean = alpha + arOne * z_lag
    vol  = sigma

    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwo(params, z):
    z_lead = z[2:len(z) - 0]
    z_lag  = z[1:len(z) - 1]
    z_llag = z[0:len(z) - 2]
    
    alpha = params[0]
    arOne = params[1]
    arTwo = params[2]
    sigma = params[3]
    
    mean = alpha + arOne*z_lag + arTwo*(z_llag ** 2)
    vol  = sigma

    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelX(params, z, ex):
    alpha = params[0]
    beta  = params[1]
    sigma = params[2]

    mean = alpha + beta * ex
    vol  = sigma

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

# 3. ARtwo:
r_par = np.array([0.8, 0.5, -0.05, 1.2])
r = np.ones(length)
for t in range(length-2):
    r[t+2] = 0.8 + 0.5 * r[t+1] - 0.05 * r[t] ** 2 + 1.2 * eps[t+2]

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
res_test = minimize(modelNorm, parNorm, args = x, method = 'L-BFGS-B')
np.round(res_test.x, 2), x_par
res_test.fun, modelNorm(res_test.x, x)

#  AR one
res_test = minimize(modelARone, parARone, args = y, method = 'L-BFGS-B')
np.round(res_test.x, 2), y_par
res_test.fun, modelARone(res_test.x, y)

#  AR two
res_test = minimize(modelARtwo, parARtwo, args = r, method = 'L-BFGS-B')
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

# EDIT PATH!!!
path = '/home/william/Dropbox/KU/K4/Python/HY2.xlsx'

xls  = pd.ExcelFile(path)
z11  = pd.read_excel(xls, 'Transformed z-score HMM_11(2)')
z01  = pd.read_excel(xls, 'Transformed z-score HMM_01(2)')
z10  = pd.read_excel(xls, 'Transformed z-score HMM_10(2)')
z11x = pd.read_excel(xls, 'Transformed z-score HMMX_11(2)')

# test: index 0 should be a number, and column name should be 'z'
z11[:2]
z01[:2]
z10[:2]
z11x[:2]

# ============================================= #

# Initial parameters
parNorm  = np.array([0.5, 0.8])
parARone = np.array([0.5, 0.4, 0.8])
parARtwo = np.array([0.5, 0.4, 0.1, 0.8])
parX     = np.array([0.5, 0.4, 0.8])

# z = np.array(np.squeeze(z11))  # Old solution; now we use pandas

# ============================================= #
# ===== Likelihood retributions =============== #
# ============================================= #

# Generate one joint table to iterate through
zs = pd.concat([z11,z01,z10,z11x], axis=1, join_axes=[z11.index])

# Contain lists containing parameters and likelihood values, resp.
parsN = np.zeros((4,2))
parsO = np.zeros((4,3))
parsT = np.zeros((4,4))
parsX = np.zeros((4,3))
llhN = np.zeros(4)
llhO = np.zeros(4)
llhT = np.zeros(4)
llhX = np.zeros(4)

for i in range(zs.shape[1]):
    res = minimize(modelNorm, parNorm, args = zs.iloc[:,i], method = 'L-BFGS-B')
    parsN[i] = res.x
    llhN[i] = res.fun

    res = minimize(modelARone, parARone, args = zs.iloc[:,i], method = 'L-BFGS-B')
    parsO[i] = res.x
    llhO[i] = res.fun

    res = minimize(modelARtwo, parARtwo, args = zs.iloc[:,i], method = 'L-BFGS-B')
    parsT[i] = res.x
    llhT[i] = res.fun

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

#    res = minimize(modelNorm, parNorm, args = zs.iloc[:,i], method = 'L-BFGS-B')
#    parsN[i] = res.x
#    llhN[i] = res.fun

