"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.
"""

import likelihoodModule as llm
import plotsModule as pltm
import EM_NM_EX as em
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from matplotlib import pyplot as plt
from numba import jit
from pandas_datareader import data as web
import genData as gd
np.set_printoptions(suppress = True)   # Disable scientific notation

@jit(nopython = True)
def stateSim(S, M, start, probs, T, u, seed = 12345):
    """
    Produces
    ---------------------------------
    Simulates M paths (vectors) of states of length T
    
    Inputs
    ---------------------------------
    S:     Scalar indicating amount of states
    M:     Scalar indicating amount of simulations
    start: Scalar indicating initial state to simulate from (state, not index)
    probs: (S x S) Transition probability matrix
    T:     Scalar simulation length, e.g. T = 12 months
    u:     Matrix (M x T) random uniform numbers between 0 and 1
    
    Returns
    ---------------------------------
    stateSim:  (M x T) M simulated paths of length T
    stateFreq: (M x S) M vectors counting frequency of each state
    """
    # Set seed number
    np.random.seed(seed)
    
    # Initialise statePaths, which will be part of final output
    statePaths = np.ones((M,T)) * start
    
    # state: Vector (S x 1) used to find the state (1, 2, ... S) which we transition to
    state      = np.ones(S)
    
    # stateFreq: Vector counting occurences of each state for each simulation
    stateFreq  = np.ones((M,S))
    
    for m in range(M):
        for t in range(1,T):
            # i defines state we are arriving from
            i = int(statePaths[m,t-1] - 1)
            for s in range(S):
                # Identifies which state we transition to
                state[s] = (np.sum(probs[:s,i]) < u[m,t] <= np.sum(probs[:s+1,i])) * (s+1)
            statePaths[m, t] = np.sum(state)
        for s in range(S):
            stateFreq[m,s] = np.sum(statePaths[m] == s + 1)
    return statePaths, stateFreq


# nopython = True not supported with our use of multivariate number generation
@jit
def returnSim(S, M, N, A, start, mu, cov, probs, T, u, seed = 12345):
    """
    Produces
    ---------------------------------
    Simulates M*N matrices (AxT) of return processes for each
    asset for a lenght of time T;
    Return processes for risky assets are generated
    by multivariate normal distribution
    
    Inputs
    ---------------------------------
    S:     Scalar indicating amount of states
    M:     Scalar indicating amount of state path simulations
    N:     Scalar indicating amount of return simulations
    A:     Scalar indicating amount of risky assets
    start: Scalar indicating initial state to simulate from (state, not index)
    mu:    (A x S) matrix of returns for each asset in each state
    cov:   (S x (A x A)) set of covariance matrices for all states
    probs: (S x S) Transition probability matrix
    T:     Scalar simulation length, e.g. T = 12 months
    u:     (M x T) matrix of random uniform numbers between 0 and 1
    
    Returns
    ---------------------------------
    returns:  (M*N x A x T) M*N simulated returns of length T for A assets
    states:   (M x T) M simulated paths of length T
    """
    np.random.seed(seed)
    states, freq = stateSim(S,M,start,probs,T,u)
    if A > 1:
        returns = np.ones((M*N,A,T))
        for m in range(M):
            for n in range(N):
                for s in range(S):
                    returns[m*N + n,:,states[m] == s + 1] = \
                        np.random.multivariate_normal(mu[:,s],cov[s],int(freq[m,s]))
    else:
        returns = np.ones((M*N,T))
        for m in range(M):
            for n in range(N):
                for s in range(S):
                    returns[m*N + n,states[m] == s + 1] = \
                        np.random.normal(mu[s],cov[s],int(freq[m,s]))
    return returns, states

"""
Perform test run by running below
---------------------------------

S = 3
M = 100
N = 1
A = 5
T = 24
Tmax = 24
start = 1
mu = np.random.normal(size = (A,S))
cov = np.array([np.cov(np.random.normal(size = 5*100).reshape(5,100)) for i in range(S)])
probs = np.array([[0.77, 0.56, 0.01],
                  [0.21, 0.90, 0.05],
                  [0.02, 0.04, 0.94]])

u = np.random.uniform(0, 1, size=(M, Tmax))

testStates, testFreq = stateSim(S,M,start,probs,T,u)
testReturns, states = returnSim(S,M,N,A,start,mu,cov,probs,T,u)
"""