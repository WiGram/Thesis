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


