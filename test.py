import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numdifftools as nd

def llh(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    mean = mu + b * z
    vol  = np.exp(v)
    #
    llh = 1 / np.sqrt(2.0 * np.pi * vol ** 2) * np.exp(-0.5 * ((x - mean) / vol) ** 2)
    #
    return - np.log(llh)

def llhSum(params, x, z):
    return np.sum(llh(params, x, z))

def firstMu(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return (x - (mu + b * z)) / v ** 2

def firstBeta(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return z * (x - (mu + b * z)) / v ** 2

def firstVar(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - 0.5 / v**2 * (1.0 - (x - (mu + b * z)) ** 2 / v ** 2)

def muMu(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - 1.0 / v ** 2

def betaBeta(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - z ** 2 / v ** 2

def varVar(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return 1.0 / v ** 4 - (x - (mu + b * z))**2 / v ** 6

def muBeta(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - z / v ** 2

def betaMu(params, x, z):
    return muBeta(params, x, z)

def muVar(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - (x - (mu + b * x)) / v ** 4

def betaVar(params, x, z):
    mu = params[0]
    b  = params[1]
    v  = params[2]
    #
    return - z * (x - (mu + b * x)) / v ** 4

def varMu(params, x, z):
    return muVar(params, x, z)

def varBeta(params, x, z):
    return betaVar(params, x, z)




T = 425

z = np.random.normal(loc = 1.0, scale = 2.0, size = T)
mu = 0.5
b  = 1.5
e = np.random.normal(size = T)

x = mu + b * z + e

plt.plot(x)
plt.show()


params = 1.0, 1.0, 0.5
args = x, z

llhSum(params, *args)
model = minimize(llhSum, params, args)
model
parEst = np.array((model.x[0], model.x[1], np.exp(model.x[2])))
parEst

# Are estimates close?
parEst - (mu, b, 1.0)

# Is likelihood value correct?
model.fun - llhSum(model.x, *args)

# Calculate standard errors numerically
hFct = nd.Hessian(llhSum)
hess = np.linalg.inv(hFct(parEst, *args))
se   = np.sqrt(np.diag(hess))

jac_fct = nd.Jacobian(llh)
jac     = jac_fct(parEst, *args)
jac     = np.transpose(jac)  # Squeeze removes a redundant dimension.
score   = np.inner(jac, jac)

# Equivalent approach to score
testScore = np.array([np.outer(jac[:,t], jac[:,t]) for t in range(T)])
np.sum(testScore, axis = 0)

robustSe = np.sqrt(np.diag(hess.dot(score).dot(hess)))
robustSe


fm = firstMu(parEst, *args)
fb = firstBeta(parEst, *args)
fv = firstVol(parEst, *args)

jac = np.array((fm, fb, fv))

score = np.array([np.outer(jac[:,t], jac[:,t]) for t in range(T)])
np.sum(score, axis = 0) / T
model.jac

hFct = nd.Hessian(llh)
hess = np.linalg.inv(hFct(parEst, *args))
se   = np.sqrt(np.diag(hess))

jacf  = nd.Jacobian(llh)
jac   = jacf(parEst, *args)
jac   = np.transpose(np.squeeze(jac, axis = 0))  # Squeeze removes a redundant dimension.
score = np.inner(jac, jac)

robustSe = np.sqrt(np.diag(hess.dot(score).dot(hess)))
robustSe

# Finding parameters
spread_opt = opt.minimize(llfArSum, par, args=spread)
estPar = spread_opt.x

# Calculate standard errors
hFct = nd.Hessian(llh)
hess = np.linalg.inv(hFct(parEst, *args))
se   = np.sqrt(np.diag(hess))

jac_fct = nd.Jacobian(llh)
jac     = jac_fct(parEst, *args)
jac     = np.transpose(jac)  # Squeeze removes a redundant dimension.
score   = np.inner(jac, jac)


robustSe = np.sqrt(np.diag(hess.dot(score).dot(hess)))
robustSe

h = model.hess_inv
h = np.linalg.inv(h)
se = np.sqrt(np.diag(h))

jac = model.jac
score = np.outer(jac, jac)
robustSe = np.sqrt(np.diag(h.dot(score).dot(h)))
robustSe