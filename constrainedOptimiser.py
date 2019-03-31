""" 
Date:    March 27th, 2019
Authors: Kristian Strand and William Gram
Subject: Constrained optimiser for portfolio weights between 0 and 1.

Suggested import:
-----------------------------
from constrainedOptimiser import constrainedOptimiser as copt

Description:
-----------------------------
Adapts scipy's optimiser to have weights constrained between 0 and 1.
The setup is not flexible beyond amounts of assets, as the problem
to be solved is trivial but required across multiple scripts.

Current issues:
-----------------------------
1. Has not yet been tested.

Functions:
-----------------------------
constrainedOptimiser(fct, par, args, A, method = SLSQP)

Imports:
-----------------------------
numpy as np
scipy.optimize as opt
"""

import numpy as np
from scipy import optimize as opt

def constrainedOptimiser(f, w, args, ApB, method = 'SLSQP'):
    """
    Constrained optimiser: weights are between 0 and 1.
    Default method set to 'SLSQP' as optimisation is constrained.
    
    Argmunets:
    -------------------------
    f       function to enter into optimiser
    w       weight parameters to optimise (could be some other quantity)
    args    returns, rf, g, T; or some other fct-specific parameters
    ApB     amount of assets incl. bank - necessary for bounds on weights
    method  SLSQP unless otherwise specified.
    """
    def check_sum(weights):
        '''
        Produces:
        -------------------------
        Returns 0 if individual weights sum to 1.0
        
        Motivation:
        -------------------------
        Applied as a constraint for opt.minimize.
        '''
        return np.sum(weights) - 1.0
    
    bnds=tuple(zip(np.zeros(ApB),np.ones(ApB)))
    cons=({'type':'eq','fun': check_sum})
    res=opt.minimize(f,w,args=args,bounds=bnds,constraints=cons,method=method)
    return res


mu1 = np.array([0.24])
mu2 = np.array([0.12])
mu  = np.concatenate((mu1,mu2))

cov = np.array([[0.14,-0.21],
                [-0.21,0.07]])

returns = np.random.multivariate_normal(mu,cov,size = 60)

w = np.array([0.3,0.4])
g = 5
args = returns, g

def expUtil(w, returns, gamma):
    W = w * np.exp(np.sum(returns, axis = 0))
    utility = np.sum(W) ** (1 - gamma)/(1-gamma)
    return - utility

def expUtil(w,returns):
    W = w * np.exp(np.sum(returns, axis = 0))
    W = np.sum(W)
    utility = W - 0.5 * W ** 2
    return -utility

args = returns
opt.minimize(expUtil, w, args = args,bounds=bnds,constraints=cons)

w  = np.random.random(size = (2,200))
ww = np.sum(w, axis = 0)
w  = w / ww
w  = w.T

utils = np.zeros(200)
for i in range(len(utils)):
    utils[i] = expUtil(w[i], returns, gamma)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(w[:,0],w[:,1],utils,c='r',marker='o')
ax.set_xlabel('High Yield allocation')
ax.set_ylabel('Russell 1000 allocation')
ax.set_zlabel('Expected Utility')
plt.show()

plt.scatter(w[:,1],utils)
plt.show()