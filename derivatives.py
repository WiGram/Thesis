import numpy as np
import numdifftools as nd
np.set_printoptions(suppress = True)   # Disable scientific notation

def firstMu(pS, returns, mu, vol):
    return pS * (returns - mu) / vol ** 2

def firstVol(pS, returns, mu, vol):
    return -0.5 * pS * (1 / vol ** 2 - (returns - mu) ** 2 / vol ** 4)

def firstPj(pSTj, pSTN, pj, pN):
    return pSTj / pj - pSTN / pN

def secondMuMu(pS, vol):
    return - sum(pS / vol ** 4)

def secondVolVol(pS, returns, mu, vol):
    return - sum( pS * ((returns - mu) ** 2 / vol ** 6 - 0.5 * 1 / vol ** 4))

def secondPjPj(pST, pj, pN):
    return -sum(pST * (1 / pj ** 2 - 1 / pN ** 2))

def secondPNPN(pST, returns, pj, pN):
    return -secondPjPj(pST, returns, pj, pN)

def secondMuVol(pS, returns, mu, vol):
    return - sum(pS * (returns - mu) / vol ** 4)

def score(pS, pST, rets, mu, vol, p):
    S = len(mu)
    N = S - 1
    T = len(rets)
    
    # One derivative for each time period t
    m = np.array([firstMu(pS[s,:], rets, mu[s], vol[s]) for s in range(S)])
    v = np.array([firstVol(pS[s,:], rets[:], mu[s], vol[s]) for s in range(S)])

    # For each column j (where we come from)
    # last row i in each column is the residual
    # Returns T-1 matrices of shape (2 x 3) -> last row is not estimated as it is a residual
    pj = np.zeros((N, S, T))
    pj[:,:,1:] = np.array([[firstPj(pST[i,j,1:], pST[N,j,1:], p[i,j], p[N,j]) for j in range(S)] for i in range(N)])
    
    score = np.array([np.concatenate((m[:,t], v[:,t], np.concatenate(pj[:,:,t]))) for t in range(T)]).T

    product = np.array([np.outer(score[:,t], score[:,t]) for t in range(T)]) / T

    return np.sqrt(np.diag(np.sum(product, axis = 0))) / np.sqrt(T)

# Only works for univariates for now
pS = pss
pST = pst
rets = returns[0]
mu = m[sims-1]
vol = v[sims-1]
p = pp[sims-1]

scoreVal = score(pS, pST, rets, mu, vol, p)
scoreVal
"""
mu   = m[m.shape[0]-1, :]
vol  = v[v.shape[0]-1, :]
p    = np.concatenate(p[p.shape[0]-1, :, :])
rets = returns[0]
pS   = pss
pST  = pst
test4 = score(pS, pST, rets, mu, vol, p)
"""


# Numerical derivative
def derivative(f,a,args,method='central',h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(a + h,args) - f(a - h,args))/(2*h)
    elif method == 'forward':
        return (f(a + h,args) - f(a,args))/h
    elif method == 'backward':
        return (f(a,args) - f(a - h,args))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def testFct(x, args):
    a = args[0]
    b = args[1]
    return - x ** 2 * a + b

args = np.array([2,3])


derivative(testFct, 2, args, method = 'central', h = 0.00001)