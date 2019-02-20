def firstMu(pS, returns, mu, vol):
    return sum(pS * (returns - mu) / vol ** 2)

def firstVol(pS, returns, mu, vol):
    return -0.5 * sum(pS * (1 / vol ** 2 - (returns - mu) ** 2 / vol ** 4))

def firstPj(pST, returns, pj, pN):
    return sum(pST * (1 / pj - 1 / pN))

def firstPN(pST, returns, pj, pN):
    return -firstPj(pST, returns, pj, pN)

def secondMuMu(pS, returns, mu, vol):
    return - sum(pS / vol ** 4)

def secondVolVol(pS, returns, mu, vol):
    return - sum( pS * ((returns - mu) ** 2 / vol ** 6 - 0.5 * 1 / vol ** 4))

def secondPjPj(pST, returns, pj, pN):
    return -sum(pST * (1 / pj ** 2 - 1 / pN ** 2))

def secondPNPN(pST, returns, pj, pN):
    return -secondPjPj(pST, returns, pj, pN)

def secondMuVol(pS, returns, mu, vol):
    return - sum(pS * (returns - mu) / vol ** 4)