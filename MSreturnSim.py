import numpy as np
from numba import jit

def returnSim4(startReg, probs, T, u):
    state_ms = np.repeat(startReg, T)
    for t in range(1,T):
        if state_ms[t-1] == 1:
            state_ms[t] = \
                (u[t] <= probs[0,0]) * 1 + \
                (sum(probs[0,:1]) < u[t] <= sum(probs[0,:2])) * 2 + \
                (sum(probs[0,:2]) < u[t] <= sum(probs[0,:3])) * 3 + \
                (sum(probs[0,:3]) < u[t] <= 1) * 4
        elif state_ms[t-1] == 2:
            state_ms[t] = \
                (u[t] <= probs[1,0]) * 1 + \
                (sum(probs[1,:1]) < u[t] <= sum(probs[1,:2])) * 2 + \
                (sum(probs[1,:2]) < u[t] <= sum(probs[1,:3])) * 3 + \
                (sum(probs[1,:3]) < u[t] <= 1) * 4
        elif state_ms[t-1] == 3:
            state_ms[t] = \
                (u[t] <= probs[2,0]) * 1 + \
                (sum(probs[2,:1]) < u[t] <= sum(probs[2,:2])) * 2 + \
                (sum(probs[2,:2]) < u[t] <= sum(probs[2,:3])) * 3 + \
                (sum(probs[2,:3]) < u[t] <= 1) * 4
        else:
            state_ms[t] = \
                (u[t] <= probs[3,0]) * 1 + \
                (sum(probs[3,:1]) < u[t] <= sum(probs[3,:2])) * 2 + \
                (sum(probs[3,:2]) < u[t] <= sum(probs[3,:3])) * 3 + \
                (sum(probs[3,:3]) < u[t] <= 1) * 4

        lenOne = sum(state_ms == 1)
        lenTwo = sum(state_ms == 2)
        lenThr = sum(state_ms == 3)
        lenFou = sum(state_ms == 4)

        return state_ms

@jit
def stateSim3(startReg, probs, T, u):
    state_ms = np.repeat(startReg, T)
    for t in range(1,T):
        if state_ms[t-1] == 1:
            state_ms[t] = \
                (u[t] <= probs[0,0]) * 1 + \
                (np.sum(probs[0,:1]) < u[t] <= np.sum(probs[0,:2])) * 2 + \
                (np.sum(probs[0,:2]) < u[t] <= 1) * 3
        elif state_ms[t-1] == 2:
            state_ms[t] = \
                (u[t] <= probs[1,0]) * 1 + \
                (np.sum(probs[1,:1]) < u[t] <= np.sum(probs[1,:2])) * 2 + \
                (np.sum(probs[1,:2]) < u[t] <= 1) * 3
        else:
            state_ms[t] = \
                (u[t] <= probs[2,0]) * 1 + \
                (np.sum(probs[2,:1]) < u[t] <= np.sum(probs[2,:2])) * 2 + \
                (np.sum(probs[2,:2]) < u[t] <= 1) * 3

        lenOne = np.sum(state_ms == 1)
        lenTwo = np.sum(state_ms == 2)
        lenThr = np.sum(state_ms == 3)

    length = np.array((lenOne, lenTwo, lenThr))

    return state_ms, length

def returnSim3(S, A, startReg, mu, cov, probs, T, u):
    state_ms, length = stateSim3(startReg, probs, T, u)

    returns = np.ones((A, T))

    for s in range(S):
        returns[:, state_ms == s + 1] = np.random.multivariate_normal(mu[:,s], cov[s], length[s]).T

    return returns

# test = returnSim3(states, assets, startReg, mu, cov, probs, mat, u)