python
import numpy as np
from numba import jit

def returnSim4(startState, probs, T, u):
    state_ms = np.repeat(startState, T)
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
        #
        lenOne = sum(state_ms == 1)
        lenTwo = sum(state_ms == 2)
        lenThr = sum(state_ms == 3)
        lenFou = sum(state_ms == 4)
        #
        length = np.array((lenOne, lenTwo, lenThr, lenFou))
        #
        return state_ms, length

@jit
def stateSim3(startState, probs, T, u):
    state_ms = np.ones(T) * startState
    for t in range(1,T):
        if state_ms[t-1] == 1:
            state_ms[t] = \
                (u[t] <= probs[0,0]) * 1 + \
                (np.sum(probs[:1,0]) < u[t] <= np.sum(probs[:2,0])) * 2 + \
                (np.sum(probs[:2,0]) < u[t] <= 1) * 3
        elif state_ms[t-1] == 2:
            state_ms[t] = \
                (u[t] <= probs[0,1]) * 1 + \
                (np.sum(probs[:1,1]) < u[t] <= np.sum(probs[:2,1])) * 2 + \
                (np.sum(probs[:2,1]) < u[t] <= 1) * 3
        else:
            state_ms[t] = \
                (u[t] <= probs[0,2]) * 1 + \
                (np.sum(probs[:1,2]) < u[t] <= np.sum(probs[:2,2])) * 2 + \
                (np.sum(probs[:2,2]) < u[t] <= 1) * 3
    #
    lenOne = np.sum(state_ms == 1)
    lenTwo = np.sum(state_ms == 2)
    lenThr = np.sum(state_ms == 3)
    #
    length = np.array((lenOne, lenTwo, lenThr))
    #
    return state_ms, length

@jit(nopython = True)
def stateSim3(startState, probs, T, u):
    state_ms = np.ones(T) * startState
    for t in range(1,T):
        state_ms[t] = \
            (state_ms[t-1] == 1) * (
                (u[t] <= probs[0,0]) * 1 + \
                (np.sum(probs[:1,0]) < u[t] <= np.sum(probs[:2,0])) * 2 + \
                (np.sum(probs[:2,0]) < u[t] <= 1) * 3
            ) + (state_ms[t-1] == 2) * (
                (u[t] <= probs[0,1]) * 1 + \
                (np.sum(probs[:1,1]) < u[t] <= np.sum(probs[:2,1])) * 2 + \
                (np.sum(probs[:2,1]) < u[t] <= 1) * 3
            ) + (state_ms[t-1] == 3) * (
                (u[t] <= probs[0,2]) * 1 + \
                (np.sum(probs[:1,2]) < u[t] <= np.sum(probs[:2,2])) * 2 + \
                (np.sum(probs[:2,2]) < u[t] <= 1) * 3
            )
    #
    lenOne = np.sum(state_ms == 1)
    lenTwo = np.sum(state_ms == 2)
    lenThr = np.sum(state_ms == 3)
    #
    length = np.array((lenOne, lenTwo, lenThr))
    #
    return state_ms, length


@jit
def returnSim3(stateSims, returnSims, S, A, startState, mu, cov, probs, T, u):
    rets = np.ones((stateSims*returnSims,A,T))
    #
    for m in range(stateSims):
        st, length = stateSim3(startState, probs, T, u[m])
        for n in range(returnSims):
            for s in range(S):
                rets[m * returnSims + n,:, st == s + 1] = np.random.multivariate_normal(mu[:,s], cov[s], length[s])
    #
    return rets


"""
T    = 12
stateSims  = 10000
returnSims = 1
u    = np.random.uniform(low = 0.0, high = 1.0, size = stateSims * returnSims * T).reshape(stateSims * returnSims, T)
rets = returnSim3(stateSims, returnSims, S, A, startState, mu, cov, probs, T, u)
test = rets[0]

plt.plot(test[0])
plt.plot(test[1])
plt.plot(test[2])
plt.plot(test[3])
plt.plot(test[4])
plt.show()
"""

