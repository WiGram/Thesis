import numpy as np
import numdifftools as nd
import derivatives as dr
import genData as gd
import EM_NM_EX as em
np.set_printoptions(suppress = True)   # Disable scientific notation

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

# ===== Monthly excess returns ===== #
# monthlyRets = monthlyRets.drop(['S&P 500', 'Gov'], axis = 1)
excessMRets = excessMRets.drop(['S&P 500'], axis = 1)
colNames = excessMRets.columns
A = len(colNames) # Assets
y = np.array(excessMRets.T) # Returns

sims = 200
S    = 3 # States
T    = len(y[0,:]) # Periods
p    = np.repeat(1.0 / S, S * S).reshape(S, S)
pS   = np.random.uniform(size = S * T).reshape(S, T)

# Multivariate
# ms, vs, ps, llh, pStar, pStarT = em.multEM(y, sims, T, S, A, p, pS)

# Univariate
m, v, pp, l, pss, pst = em.uniEM(y[0], sims, T, S, p, pS)

# Only works for univariates for now
scoreVal = score(pss, pst, y[0], m[sims-1], v[sims-1], pp[sims-1])
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


args = np.array([2,3])


derivative(testFct, 2, args, method = 'central', h = 0.00001)