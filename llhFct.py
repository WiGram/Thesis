"""
Date:    February 26th, 2019
Authors: Kristian Strand and William Gram
Subject: Parameter estimation using numerical differentiation

Description:
Defining the log-likelihood function as a function of all
parameters directly. This will then be numerically differentiated
using either numdifftools, if it works, or custom differentiation
functions.
"""

from numba import jit
import numpy as np

@jit(nopython = True)
def density(d, dR, covm, A):
        return 1 / np.sqrt( (2 * np.pi) ** A * d) * np.exp(-0.5 * dR.dot(np.linalg.inv(covm)).dot(dR))

@jit(nopython = True)
def densityUni(dR, var):
        return 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * dR ** 2 / var)


@jit
def llhFct(params, y, pS, pST):
    S = pS.shape[0]  # has shape 5,3
    A = y.shape[0]  # see above.
    T = pS.shape[1] # has shape 3,3,425
    
    mu = params[0]
    covm = params[1]
    p = np.zeros((S,S))
    p[:S-1,:] = params[2]

    p[S-1, :] = 1 - np.sum(p[:S-1,:], axis = 0)
    
    """
    d:   determinant
    dR:  demeaned returns
    k:   Some constant containing c and 2pi (see mid of page 14, lecture note 5)
    sum_pIJ: First sum in (V.13) page 13 of lecture note 5
    sum_pJ:  Second sum in (V.13) page 13 of lecture note 5
    
    """

    d  = np.linalg.det(covm) # returns [det1, det2, ..., detN], N: amount of states
    dR = np.zeros((S,A,T))
    for s in range(S):
        for a in range(A):
            dR[s,a,:] = y[a,:] - mu[a,s]

    f  = np.zeros((S, T))
    for s in range(S):
        for t in range(T):
            f[s, t] = density(d[s], dR[s, :, t], covm[s], A)
    
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    
    # first sum (V.13), page 13
    sum_pIJ = 0
    for i in range(S):
        for j in range(S):
            sum_pST = np.sum(pST[j, i, 1:])
            sum_pIJ += np.log(p[j, i]) * sum_pST

    # Second sum (V.13), page 13
    sum_pJ = 0
    for j in range(S):
        sum_pJ += np.sum(pS[j, :] * np.log(f[j, :]))

    return k + sum_pIJ + sum_pJ

@jit
def llhUniFct(params, y, pS, pST):
    S = pS.shape[0]  # has shape 5,3
    T = len(y) # has shape 3,3,425

    mu = params[:S]
    covm = params[S:2*S]
    p = np.zeros((S,S))
    p[:S-1,:] = params[2*S:].reshape(S-1, S)
    p[S-1, :] = 1 - np.sum(p[:S-1,:], axis = 0)
    
    """
    d:   determinant
    dR:  demeaned returns
    k:   Some constant containing c and 2pi (see mid of page 14, lecture note 5)
    sum_pIJ: First sum in (V.13) page 13 of lecture note 5
    sum_pJ:  Second sum in (V.13) page 13 of lecture note 5
    
    """

    dR = np.zeros((S,T))
    for s in range(S):
        dR[s,:] = y - mu[s]

    f  = np.zeros((S, T))
    for s in range(S):
        f[s, :] = densityUni(dR[s,:], covm[s])
    
    k = -0.5 * (np.log(2 * np.pi) + 1.0)  # the constant 'c' is set to 1.0
    
    # first sum (V.13), page 13
    sum_pIJ = 0
    for i in range(S):
        for j in range(S):
            sum_pST = np.sum(pST[j, i, 1:])
            sum_pIJ += np.log(p[j, i]) * sum_pST

    # Second sum (V.13), page 13
    sum_pJ = 0
    for s in range(S):
        sum_pJ += np.sum(pS[s, :] * np.log(f[s, :]))

    return k + sum_pIJ + sum_pJ

