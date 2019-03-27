import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numdifftools as nd
np.set_printoptions(suppress = True)   # Disable scientific notation

def llh(params, x):
    mu  = params[0]
    var = params[1] # Variance !!!
    #
    llh = 1 / np.sqrt(2.0 * np.pi * var) * np.exp(-0.5 * (x - mu) ** 2 / var)
    #
    return -np.log(llh)

def llhSum(params, x):
    return np.sum(llh(params, x))

def firstMu(params, x):
    mu  = params[0]
    var = params[1]
    #
    return (x - mu) / var

def firstVar(params, x):
    mu  = params[0]
    var = params[1]
    #
    "Pay attention!: var = vol^2"
    return -0.5 / var + 0.5 * (x - mu) ** 2 / var ** 2

def muMu(params, x):
    mu  = params[0]
    var = params[1]
    #
    return np.repeat(- 1.0 / var, len(x))

def muVar(params, x):
    mu  = params[0]
    var = params[1]
    #
    return - (x - mu) / var ** 2

def varVar(params, x):
    mu  = params[0]
    var = params[1]
    #
    return 0.5 / var ** 2 - (x - mu)**2 / var ** 3

# GENERATING MODEL DATA
T = 5
e = np.random.normal(size = T)
mu = 0.5
x = mu + e

# ESTIMATING PARAMETERS
params = 1.0, 0.5
model = minimize(llhSum, params, x)
parEst = model.x
parEst

# GENERATING HESSIAN
hFct = nd.Hessian(llhSum)
hess = hFct(parEst, x)
hInv = np.linalg.inv(hess)
se   = np.sqrt(np.diag(hInv))
se

# GENERATING JACOBIAN AND SCORE
jac_fct = nd.Jacobian(llh)
jac     = jac_fct(parEst, x)[:,:20]
jac     = np.transpose(jac)  # Squeeze removes a redundant dimension.
score   = np.inner(jac, jac)
score

# COMPUTING ROBUSET SE
robustSe = np.sqrt(np.diag(hInv.dot(score).dot(hInv)))
robustSe
se

fm = firstMu(parEst, x)
fv = firstVar(parEst, x)

anJac = np.array((fm, fv))
anScore = np.inner(anJac, anJac)
anScore
score

mm = muMu(params, x)
mv = muVar(params, x)
vv = varVar(params, x)

anHess = np.array([[mm, mv], [mv, vv]])
anHess = -np.sum(anHess, axis = 2)
anHess
hess

hAI = np.linalg.inv(anHess)
anSe = np.sqrt(np.diag(hAI))
anSe
se

anRobSe = np.sqrt(np.diag(hAI.dot(anScore).dot(hAI)))
anRobSe/2
robustSe