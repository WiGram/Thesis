"""
Description:
-----------------------------
Specify parameters in univariate and multivariate GARCH-models,
in addition to simulating these processes based on derived
parameters.

Univariate model:
-----------------------------
x_t = s_t*e_t, e_t~N(0,1)
s_t^2 = a + b*x_(t-1)^2 + c*s_(t-1)^2

Imports:
-----------------------------
numpy as np
numba.jit
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import scipy.optimize as opt

@jit(nopython = True)
def simUniGARCH(a,b,c,e):
    """
    Parameters
    -----------------------------
    a       Scalar indicating constant in s_t^2
    b       Scalar indicating AR effect
    c       Scalar indicating ARCH effect
    e       Standard normal process
    """
    T = len(e)
    x = np.ones(T)
    s = np.ones(T)
    for t in range(T-1):
        s[t+1] = np.sqrt(a + b*x[t]**2 + c*s[t]**2)
        x[t+1] = s[t+1] * e[t+1]
    return x,s

def estUniGARCH(theta,x,s):
    """
    theta = a,b,c
    args  = x,s
    """
    a, b, c = np.exp(theta)
    
    T = len(x)
    
    x_lead = x[1:]
    x_lag  = x[:T-1]
    s_lag  = s[:T-1]
    
    mean = 0.
    var  = a + b*x_lag**2 + c*s_lag**2
    
    loglikelihoodContributions = -0.5*(np.log(2.*np.pi*var) + (x_lead-mean)**2 / var)
    
    return -np.sum(loglikelihoodContributions)

T = 120
e = np.random.normal(loc = 0.0, scale = 1.0, size = T)

a = 0.5
b = 0.2
c = 0.3
M = 100000
# Simulation
x = np.zeros((M,T))
for i in range(M):
    x[i],s = simUniGARCH(a,b,c,e)

plt.plot(x[1])
plt.show()

theta = (a,b,c)
args = x,s

thetaEst = np.exp((a,b,c))
results = opt.minimize(estUniGARCH,thetaEst,args=args,method='Nelder-Mead')
results.fun, estUniGARCH(theta,*args)
np.exp(results.x)

import pandas as pd

def gjr_garch_likelihood(parameters, data, sigma2, out=None):
    ''' Returns negative log-likelihood for GJR-GARCH(1,1,1) model.'''
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]
    T = np.size(data,0)
    eps = data - mu
    # Data and sigma2 are T by 1 vectors
    for t in range(1,T):
        sigma2[t] = (omega + alpha * eps[t-1]**2 \
            + gamma * eps[t-1]**2 * (eps[t-1]<0) + beta * sigma2[t-1])
    
    logliks = 0.5*(np.log(2*np.pi) + np.log(sigma2) + eps**2/sigma2)
    loglik = np.sum(logliks)
    if out is None:
        return loglik
    else:
        return loglik, logliks, np.copy(sigma2)


def gjr_constraint(parameters, data, sigma2, out=None):
    ''' Constraint that alpha+gamma/2+beta<=1'''
    
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]
    
    return np.array([1-alpha-gamma/2-beta])


def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5*np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = np.size(theta,0)
    h = np.diag(h)
    fp = np.zeros(K)
    fm = np.zeros(K)
    for i in range(K):
        fp[i] = fun(theta+h[i], *args)
        fm[i] = fun(theta-h[i], *args)
        fpp = np.zeros((K,K))
        fmm = np.zeros((K,K))
    
    for i in range(K):
        for j in range(i,K):
            fpp[i,j] = fun(theta + h[i] + h[j], *args)
            fpp[j,i] = fpp[i,j]
            fmm[i,j] = fun(theta - h[i] - h[j], *args)
            fmm[j,i] = fmm[i,j]
        hh = (np.diag(h))
        hh = hh.reshape((K,1))
        hh = hh @ hh.T
        H = np.zeros((K,K))
    
    for i in range(K):
        for j in range(i,K):
            H[i,j] = (fpp[i,j] - fp[i] - fp[j] + f \
                + f - fm[i] - fm[j] + fmm[i,j])/hh[i,j]/2
        H[j,i] = H[i,j]
    return H

startingVals = np.array([x.mean(), x.var() * .01, .03, .09, .90])

finfo = np.finfo(np.float64)
bounds = [(-10*x.mean(),10*x.mean()),
          (finfo.eps,2*x.var() )
          ,(0.0,1.0),(0.0,1.0),(0.0,1.0)]

T = x.shape[0]
sigma2 = np.ones(T) * x.var()
# Pass a NumPy array, not a pandas Series
args = (np.asarray(x), sigma2)
estimates = opt.fmin_slsqp(gjr_garch_likelihood, startingVals, \
    f_ieqcons=gjr_constraint, bounds = bounds, \
    args = args)

loglik, logliks, sigma2final = gjr_garch_likelihood(estimates, x, \
    sigma2, out=True)

step = 1e-5 * estimates
scores = np.zeros((T,5))
for i in range(5):
    h = step[i]
    delta = np.zeros(5)
    delta[i] = h
    
    loglik, logliksplus, sigma2 = gjr_garch_likelihood(estimates + delta, \
        np.asarray(x), sigma2, out=True)
    loglik, logliksminus, sigma2 = gjr_garch_likelihood(estimates - delta, \
        np.asarray(x), sigma2, out=True)
    
    scores[:,i] = (logliksplus - logliksminus)/(2*h)

I = (scores.T @ scores)/T

J = hessian_2sided(gjr_garch_likelihood, estimates, args)
J = J/T
Jinv = np.mat(np.linalg.inv(J))
vcv = Jinv*np.mat(I)*Jinv/T
vcv = np.asarray(vcv)

output = np.vstack((estimates,np.sqrt(np.diag(vcv)),estimates/np.sqrt(np.diag(vcv)))).T
print('Parameter Estimate Std. Err. T-stat')
param = ['mu','omega','alpha','gamma','beta']
for i in range(len(param)):
    print('{0:<11} {1:>0.6f} {2:0.6f} {3: 0.5f}'.format(param[i],
    output[i,0], output[i,1], output[i,2]))

fig = plt.figure()
ax = fig.add_subplot(111)
volatility = pd.DataFrame(np.sqrt(252 * sigma2))
ax.plot(volatility.index,volatility)
ax.autoscale(tight='x')
fig.autofmt_xdate()
fig.tight_layout(pad=1.5)
ax.set_ylabel('Volatility')
ax.set_title('FTSE Volatility (GJR GARCH(1,1,1))')
plt.show()


"""

a, b, c = 1.0, 0.1, 0.8
e = np.random.normal(size = 10000)
X,S = simUniGARCH(a,b,c,e)

X = X[1000:] # Drop burn in
X = X / np.std(X) # Normalising X
"""

2*1