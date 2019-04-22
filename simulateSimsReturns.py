"""
Date:    March 12th, 2019
Authors: Kristian Strand and William Gram
Subject: Choosing optimal portfolio weights

Description:
We intend to find portfolio weights from a CRRA quadratic
utility function.
"""

import numpy as np
from numba import jit
np.set_printoptions(suppress=True)   # Disable scientific notation


@jit(nopython=True)
def stateSim(S, M, start, probs, T, u, seed=12345):
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
    statePaths = np.ones((M, T)) * start

    # state: Vector (S x 1) used to find the state (1, 2, ... S)
    # which we transition to:
    state = np.ones(S)

    # stateFreq: Vector counting occurences of each state for each simulation
    stateFreq = np.ones((M, S))

    for m in range(M):
        for t in range(1, T):
            # i defines state we are arriving from
            i = int(statePaths[m, t-1] - 1)
            for s in range(S):
                # Identifies which state we transition to
                state[s] = (np.sum(probs[:s, i]) < u[m, t] <= np.sum(probs[:s+1, i]))*(s+1)
            statePaths[m, t] = np.sum(state)
        for s in range(S):
            stateFreq[m, s] = np.sum(statePaths[m] == s + 1)
    return statePaths, stateFreq


# nopython = True not supported with our use of multivariate number generation
@jit
def returnSim(S, M, N, A, start, mu, cov, probs, T, u, seed=12345):
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
    states, freq = stateSim(S, M, start, probs, T, u)
    if A > 1:
        returns = np.ones((M*N, A, T))
        for m in range(M):
            for n in range(N):
                for s in range(S):
                    returns[m*N + n, :, states[m] == s + 1] = \
                        np.random.multivariate_normal(
                            mu[:, s], cov[s], int(freq[m, s])
                        )
    else:
        returns = np.ones((M*N, T))
        for m in range(M):
            for n in range(N):
                for s in range(S):
                    returns[m*N + n, states[m] == s + 1] = \
                        np.random.normal(mu[s], cov[s], int(freq[m, s]))
    return returns, states


@jit
def returnSimUniMix(S, SPs, N, start, V, M, AR, C, P, T, paths, seed=12345):
    """
    Produces
    ---------------------------------
    Simulates M*N matrices (AxT) of return processes for each
    asset for a lenght of time T;
    Return processes for risky assets are generated
    by multivariate normal distribution.
    Each asset has its own state path.
    The covariances are static (variances switch)

    Important!!!
    ---------------------------------
    The covariance matrix is not guaranteed to be positive semidefinite.

    We have applied an unconventional fix to this, where negative eigen-
    values have been coerced positive, albeit small.

    To be precise, by eigendecomposition we have extracted the eigenvectors and
    eigenvalues, and after correcting eigenvalues we reconstruct a positive
    semidefinite covariance matrix.

    Inputs -- (A indicates no. assets)
    ---------------------------------
    S:     Scalar indicating amount of states
    SPs:   Scalar indicating amount of state path simulations
    N:     Scalar indicating amount of return simulations
    start: Scalar indicating initial state to simulate from (state, not index)
    V:     (A x S) State switching variances
    M:     (A x S) matrix of returns for each asset in each state
    AR:    (A x S) matrix of autoregressive coefs for each asset and state
    C:     (A x A) State invariant (i.e. static) covariance matrix
    P:     (A x (S x S)) Transition probability matrices for each asset
    T:     Scalar simulation length, e.g. T = 12 months
    paths: (A x (SP x T)) SP amount of simulated state paths for each asset.

    Returns
    ---------------------------------
    returns:  (M*N x A x T) M*N simulated returns of length T for A assets
    """
    keys = list(M.keys())
    k = keys[0]
    A = len(keys)
    np.random.seed(seed)
    returns = np.zeros((SPs*N, A, T))
    corrections = 0
    for m in range(SPs):
        for n in range(N):
            for t in range(T-1):
                var = [V[k][int(paths[k][m*N+n, t]-1)] for k in keys]
                C[np.diag_indices_from(C)] = var
                if np.any(np.linalg.eigvals(C) < 0):
                    eigen_values, eigen_vector = np.linalg.eig(C)
                    eigen_values[eigen_values < 0] = 0.0001
                    D = np.diag(eigen_values)
                    C = np.round(eigen_vector @ D @ eigen_vector.T, 4)
                    corrections += 1
                ar = [AR[k][int(paths[k][m*N+n, t]-1)] for k in keys]
                mu = [M[k][int(paths[k][m*N+n, t]-1)] for k in keys]
                mean = mu + ar * returns[m*N+n, :, t]
                returns[m*N + n, :, t+1] = \
                    np.random.multivariate_normal(mean, C)
    print(corrections)
    return returns

@jit
def returnARSim(S, M, N, A, start, mu, ar, cov, probs, T, u, seed=12345):
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
    ar:    (A x S) matrix of AR(1) coefficients (i.e. no cross-autocorrelation)
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
    states, freq = stateSim(S, M, start, probs, T, u)
    if A > 1:
        returns = np.zeros((M*N, A, T))
        for m in range(M):
            for n in range(N):
                for t in range(T-1):
                    s = int(states[m*N+n, t] - 1)  # state index
                    mean = mu[:, s] + ar[:, s] * returns[m*N+n, :, t]
                    returns[m*N + n, :, t+1] = \
                        np.random.multivariate_normal(mean, cov[s])
    else:
        returns = np.zeros((M*N, T))
        for m in range(M):
            for n in range(N):
                for t in range(T-1):
                    s = int(states[m*N + n, t] - 1)
                    mean = mu[s] + ar[s] * returns[m*N + n, t]
                    returns[m*N + n, t+1] = \
                        np.random.normal(mean, cov[s])
    return returns, states


if __name__ == '__main__':
    # Perform test run by running below
    # ---------------------------------

    S = 3
    M = 100
    N = 1
    A = 5
    T = 24
    Tmax = 24
    start = 1
    mu = np.random.normal(size=(A, S))
    cov = np.array([np.cov(
        np.random.normal(size=5*100).reshape(5, 100)
    ) for i in range(S)])
    probs = np.array([[0.77, 0.56, 0.01],
                      [0.21, 0.90, 0.05],
                      [0.02, 0.04, 0.94]])

    u = np.random.uniform(0, 1, size=(M, Tmax))

    testStates, testFreq = stateSim(S, M, start, probs, T, u)
    testReturns, states = returnSim(S, M, N, A, start, mu, cov, probs, T, u)
