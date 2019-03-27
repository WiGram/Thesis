"""
Date:    March 13th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
Computes normalised portfolio weights
"""

import numpy as np
from numba import jit
np.set_printoptions(suppress = True)   # Disable scientific notation

@jit(nopython = True)
def pfWeights(w):
    """
    Produces
    -----------------------------------
    A set of normalised weights
    
    Inputs
    -----------------------------------
    w:    (A+1 x W) matrix of random normal numbers
    
    Outputs
    -----------------------------------
    normedW:  (A+1 x W) matrix of weights where columns sum to 1
    """
    
    # Counting how many assets can be invested in
    ApB = w.shape[0]
    
    # Counting how many sets of portfolio weights are to be normalised
    W = w.shape[1]
    
    # Initialise
    normedW = np.ones((ApB, W))
    for i in range(W):
        normedW[:,i] = w[:,i] / np.sum(w[:,i])
    #
    return normedW

"""
A = 5
ApB = A + 1
W = 1000
w = np.random.random(size = (ApB, W))

testWeights = pfWeights(w)

testSum = np.sum(testWeights, axis = 0)
testSum.min()
testSum.max()
testSum.mean()
"""