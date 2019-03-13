"""
Date:    February 2nd, 2019
Authors: Kristian Strand and William Gram
Subject: Finding optimal portfolio weights

Description:
Script combines
 * pfWeightsModule
 * simulateSimsReturns
 * expUtilModule
To produce expected utility, and retrieve optimal weights
"""

import numpy as np
from numba import jit
from pfWeightsModule import pfWeights
from simulateSimsReturns import returnSim
from expUtilModule import expectedUtility
np.set_printoptions(suppress = True)   # Disable scientific notation


@jit
def findOptimalWeights(M,N,W,T,S,A,rf,G,start,mu,cov,probs,u,w):
    """
    Produces
    ---------------------------------------------
    By combining three modules, we find expected utilities
    for different combinations of portfolio allocations and
    we find the portfolio allocation that optimises utility.
    
    Inputs
    ---------------------------------------------
    M:     Scalar indicating how many state paths are simulated
    N:     Scalar indicating how many sets of return paths are simulated
    W:     Scalar indicating how many portfolio allocations are possible
    T:     Scalar indicating how many periods have been simulated
    S:     Scalar indicating how many states are considered
    A:     Scalar indicating amount of assets considered
    rf:    Scalar for risk-free rate
    R:     (M*N x (A x T)) matrix of M*N simulated returns for A assets of T periods
    G:     Scalar GAMMA indicating degree of risk aversion
    start: Scalar indicating initial state to simulate from
    mu:    (A x S) matrix of returns for each asset in each state
    cov:   (S x (A x A)) set of covariance matrices for all states
    probs: (S x S) Transition probability matrix
    u:     (M x T) matrix of random uniform numbers between 0 and 1
    w:     (A+1 x W) matrix of random normal numbers between 0 and 1
    
    Modules called
    ---------------------------------------------
    pfWeightsModule
    simulateSimsReturns
    expUtilModule
    
    Returns
    ---------------------------------------------
    eU:       (Wx1) Vector of expected utility values
    uMax:     Scalar of the maximal expected utility
    uArgMax:  Scalar of the index of the maximal expected utility
    wMax:     Vector of weights that provide maximal expected utility
    R:        (M*N x A x T) matrix; M*N simulated returns of length T for A assets
    states:   (M x T); M simulated paths of length T
    wM:       (A+1 x W) matrix of weights where columns sum to 1
    """
    
    # Produce M*N simulated returns of length T for A assets; (M*N x A x T)
    R, states  = returnSim(S,M,N,A,start,mu,cov,probs,T,u)
    
    # Produce (A+1 x M) matrix of weights where columns sum to 1
    wM = pfWeights(w)
    
    ApB = A + 1
    
    # Produce final output
    eU,uMax,uArgMax,wMax = expectedUtility(M,N,W,T,rf,R,wM,G,ApB)
    
    return eU, uMax, uArgMax, wMax, R, states, wM


"""
# Test the function by running the below code
# -----------------------------------------------
from pfWeightsModule import pfWeights
import simulateSimsReturns as ssr

# -----------------------
# TO RUN ssr.returnSim
S = 3
A = 5
M = 100
N = 1
T = 12
start = 1
mu = np.random.normal(size = (A,S))
cov = np.array([np.cov(np.random.normal(size = 5*100).reshape(5,100)) for i in range(S)])
probs = np.array([[0.77, 0.56, 0.01],
                  [0.21, 0.90, 0.05],
                  [0.02, 0.04, 0.94]])

u = np.random.uniform(0,1,M * T).reshape(M,T)

ApB = A + 1
W = 1000
weights = np.random.random(size = (ApB, W))
# ----------------------

rf = 0.19

G = 5
ApB = ApB

eU,uMax,uArgMax,wMax,rets,states,allWeights=findOptimalWeights(M,N,W,T,S,A,rf,G,start,mu,cov,probs,u,weights)

"""