"""
Date:    February 2nd, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
This script takes a CRRA utility function and
outputs expected utility.
"""

import numpy as np
from numba import jit
np.set_printoptions(suppress = True)   # Disable scientific notation


@jit(nopython = True)
def expectedUtility(w,M,N,T,rf,R,G):
    """
    Produces
    ---------------------------------------------
    Vector of expected utilities for each possible
    portfolio allocation determined by the amount
    of sets of weights simulated (W).
    
    Inputs
    ---------------------------------------------
    M:    Scalar indicating how many state paths are simulated
    N:    Scalar indicating how many sets of return paths are simulated
    T:    Scalar indicating how many periods have been simulated
    rf:   Scalar for risk-free rate
    R:    (M*N x (A x T)) matrix of M*N simulated returns for A assets of T periods
    w:    ((A+1) x 1) vector of normalised weights
    G:    Scalar GAMMA indicating degree of risk aversion
    
    Returns
    ---------------------------------------------
    eU:       Scalar of expected utility
    uMax:     Scalar of the maximal expected utility
    uArgMax:  Scalar of the index of the maximal expected utility
    wMax:     Vector of weights that provide maximal expected utility
    """
    
    """    
    # Transformation of weights so they are in the interval [0,1]
    w = np.exp(w)/(1+np.exp(w))
    
    # Normalise to one
    w /= np.sum(w)
    """
    
    # Precompute risk aversion parameter
    RA = 1. - G
    
    # Convert returns to decimal representation
    R  = R / 100.0
    rf = rf / 100.0
    
    # Compute weighted cumulated return from risk-free asset (Scalar)
    cRF = w[0] * np.exp(T * rf)
    """
    (1) Returns are compounded across time - sum must be along columns: axis = 1
    (2) We are testing all W different portfolio weights
    """
    
    # Compute weighted cumulated return from risky assets
    cRR = w[1:] * np.exp(np.sum(rf + R, axis = 2))
    
    return -np.sum((cRF + np.sum(cRR, axis = 0) ) ** RA / RA) / (M * N)

"""
# Test the function by running the below code
# -----------------------------------------------
import simulateSimsReturns as ssr
import scipy.optimize as opt

# -----------------------
# TO RUN ssr.returnSim
S = 3
A = 5
ApB = A + 1
M = 100
N = 1
T = 12
G = 5
start = 1
mu = np.random.normal(size = (A,S))
cov = np.array([np.cov(np.random.normal(size = (5,100))) for i in range(S)])
probs = np.array([[0.77, 0.56, 0.05],
                  [0.16, 0.88, 0.09],
                  [0.07, 0.06, 0.86]])
w = np.random.random(ApB)
rf = 0.19
# ----------------------

np.random.seed(12345)
u = np.random.uniform(0,1,(M,T))
R, states = ssr.returnSim(S, M, N, A, start, mu, cov, probs, T, u)

args = M,N,T,rf,R,G

# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1.0

# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})

# 0-1 bounds for each weight
bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))

# Initial Guess (equal distribution)
w = np.random.random(ApB)

x = opt.minimize(expectedUtility, w, args, bounds = bounds, constraints = cons).x
x

"""