python
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
from scipy import stats
from scipy.optimize import minimize
from matplotlib import pyplot as plt
np.set_printoptions(suppress = True)   # Disable scientific notation

# ============================================= #
# ===== Model functions ======================= #
# ============================================= #

def testStat(lStd, lTest):
    return - 2 * (lStd - lTest)

def pValue(testStat, type = 'normal'):
    if type == 'normal':
        return ValueError('WIP')

def densityFct(z, mean, vol):
    return 1 / np.sqrt(2.0 * np.pi * vol ** 2) * np.exp(-0.5 * (z - mean)**2 / vol ** 2)

def modelStd(z):
    return - np.sum(  np.log(densityFct(z, 0.0, 1.0)))

def modelNorm(params, z):
    alpha = params[0]  # alpha
    gamma  = params[1]  # sigma
    #
    mean = alpha
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z, mean, vol))  )

def modelARone(params, z):
    z_lag  = z.shift(1)[1:]
    z_lead = z[1:]
    #
    alpha = params[0]
    gamma = params[1] # sigma = exp(gamma) <= ensures sigma > 0
    arOne = params[2]
    # 
    mean = alpha + arOne * z_lag
    vol  = np.exp(gamma)
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwo(params, z):
    z_llag = z.shift(2)[2:]
    z_lag  = z.shift(1)[2:]
    z_lead = z[2:]
    # 
    alpha = params[0]
    gamma = params[1]
    arOne = params[2]
    arTwo = params[3]
    #
    mean = alpha + arOne*z_lag + arTwo*z_llag
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwoS(params, z):
    z_llag = z.shift(2)[2:]
    z_lag  = z.shift(1)[2:]
    z_lead = z[2:]
    # 
    alpha  = params[0]
    gamma  = params[1]
    arOne  = params[2]
    arTwo  = params[3]
    arOneS = params[4]
    arTwoS = params[5]
    #
    mean = alpha + arOne*z_lag + arTwo*z_llag + arOneS*(z_lag**2) + arTwoS*(z_llag**2)
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log( densityFct(z_lead, mean, vol) )  )

def modelX(params, z, ex):
    alpha = params[0]
    sigma = params[1]
    beta  = params[2]
    #
    mean = alpha + beta * ex
    vol  = np.exp(sigma)
    #
    return - np.sum(  np.log(densityFct(z, mean, vol))   )

def modelARX(params, z, ex):
    z_lag  = z.shift(1)[1:]
    z_lead = z[1:]    
    #
    alpha = params[0]
    sigma = params[1]
    beta  = params[2]
    arOne = params[3]
    #
    mean = alpha + beta * ex + arOne * z_lag
    vol  = np.exp(sigma)
    #
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))   )


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
zs.columns = ['z11','z01','z10','z11x']


path = '/home/william/Dropbox/KU/K4/Python/HY3.xlsx'
xls  = pd.ExcelFile(path)
zs   = pd.read_excel(xls, 'Sheet5')
z11  = zs.iloc[:,0]
z01  = zs.iloc[:,1]
z10  = zs.iloc[:,2]
z11x = zs.iloc[:,3]
z11v = zs.iloc[:,4]
z11xv= zs.iloc[:,5]
# test: index 0 should be a number, and column name should be 'z'
z11[:2]
z01[:2]
z10[:2]
z11x[:2]
"""

path = '/home/william/Dropbox/Thesis/Excel/3-HY-2-state-z-score-computations.xlsx'
xls  = pd.ExcelFile(path)
zs   = pd.read_excel(xls, 'Sheet1')
A = len(zs.columns)

div  = pd.read_excel('/home/william/Dropbox/KU/K4/Python/DivYield.xlsx', 'Monthly')
div  = div.iloc[:,1]


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

testTitles = np.array(['Z(1,1)','Z(0,1)','Z(1,0)','Z(1,1)(X)'])
fig, axes = plt.subplots(nrows = 2,ncols = 2,
                         sharex = True,
                         sharey = True,
                         figsize = (15,6))
pseudoRes = np.array([zs.iloc[:,j] for j in range(zs.shape[1])])
for ax, title, y in zip(axes.flat, testTitles, pseudoRes):
    ax.plot(range(zs.shape[0]), y ** 2)
    ax.set_title(title)
    ax.grid(False)

plt.show()

# ============================================= #


# ============================================= #
# ===== Likelihood retributions =============== #
# ============================================= #

# Initial parameters: Values are guided by sporadic minimisation tests
llh = pd.DataFrame(np.zeros((A,5)), 
                   columns = ['Standard','Normal','AR(1)','Ext. AR(2)','Exog.'], 
                   index = zs.columns)

t_stat = pd.DataFrame(np.zeros((A,4)), 
                      columns = ['Normal','AR(1)','Ext. AR(2)','Exog.'], 
                      index = zs.columns)

p_val = pd.DataFrame(np.zeros((A,4)), 
                      columns = ['Normal','AR(1)','Ext. AR(2)','Exog.'], 
                      index = zs.columns)

parsN = pd.DataFrame(np.zeros((A,2)), 
                    columns = ['Mean','Variance'], 
                    index = zs.columns)

parsO = pd.DataFrame(np.zeros((A,3)), 
                    columns = ['Mean','Variance','AR(1)'], 
                    index = zs.columns)

parsT = pd.DataFrame(np.zeros((A,6)), 
                    columns = ['Mean','Variance','AR(1)','AR(2)','Sq. AR(1)','Sq. AR(2)'], 
                    index = zs.columns)

parsX = pd.DataFrame(np.zeros((A,3)), 
                    columns = ['Mean','Variance','Beta_X'], 
                    index = zs.columns)

# Solver is sensitive to suggestions on these parameters
parNorm   = np.array([-0.04, 1.0])
parARone  = np.array([0.04, 1.0, 0.1])
parARtwoS = np.array([0.04, 1.0, 0.05, 0.04, 0.03, 0.02])
parX      = np.array([0.5, 1.0, 0.4])

# Contain lists containing parameters and likelihood values, resp.

method = 'L-BFGS-B'
for i in range(zs.shape[1]):
    # Standard computation
    llh.iloc[i,0]    = -modelStd(zs.iloc[:,i])
    # Normal test
    res = minimize(modelNorm, parNorm, args = zs.iloc[:,i], method = method)
    parsN.iloc[i,:2] = np.hstack(   ( res.x[0], np.exp(res.x[1]))  )
    llh.iloc[i,1]    = -res.fun
    t_stat.iloc[i,0] = testStat(llh.iloc[i,0], llh.iloc[i,1])
    p_val.iloc[i,0]  = 1 - stats.chi2.cdf(t_stat.iloc[i,0], 2)
    # AR(1) test
    res = minimize(modelARone, parARone, args = zs.iloc[:,i], method = method)
    parsO.iloc[i,:]  = np.hstack(   ( res.x[0], np.exp(res.x[1]), res.x[2:] )   )
    llh.iloc[i,2]    = -res.fun
    t_stat.iloc[i,1] = testStat(llh.iloc[i,0], llh.iloc[i,2])
    p_val.iloc[i,1]  = 1 - stats.chi2.cdf(t_stat.iloc[i,1], 3)
    # AR(2) test with quadratic second lag
    res = minimize(modelARtwoS, parARtwoS, args = zs.iloc[:,i], method = method)
    parsT.iloc[i,:]  = np.hstack(   ( res.x[0], np.exp(res.x[1]), res.x[2:] )   )
    llh.iloc[i,3]    = -res.fun
    t_stat.iloc[i,2] = testStat(llh.iloc[i,0], llh.iloc[i,3])
    p_val.iloc[i,2]  = 1 - stats.chi2.cdf(t_stat.iloc[i,2], 6)
    # Normal with exogenous regressor
    args = zs.iloc[:,i], div
    res = minimize(modelX, parX, args = args, method = method)
    parsX.iloc[i,:]  = np.hstack(   ( res.x[0], np.exp(res.x[1]), res.x[2:] )   )
    llh.iloc[i,4]    = -res.fun
    t_stat.iloc[i,3] = testStat(llh.iloc[i,0], llh.iloc[i,4])
    p_val.iloc[i,3]  = 1 - stats.chi2.cdf(t_stat.iloc[i,3], 3)



parsN.round(4)
parsO.round(4)
parsT.round(4)
parsX.round(4)
llh.round(2)
t_stat.round(2)
p_val.round(4)