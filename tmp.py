import pyfolio as pf
import empyrical as ep
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numba import jit
from scipy import optimize as opt
import time


prices = pd.read_csv(
            # '/home/william/Dropbox/Thesis/mthReturns.csv',
            'C:/Users/willi/Dropbox/Thesis/mthReturns.csv',
            index_col=0,
            header=0)
prices = prices.drop(['SPXT'], axis=1)
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()
colNames = ['High Yield',
            'Investment Grade',
            'Commodities',
            'Russell 2000',
            'Russell 1000']
prices.columns = colNames
prices.head()

prices.plot()
plt.show()

rf_data = pd.read_csv(
            # '/home/william/Dropbox/Thesis/rf.csv',
            'C:/Users/willi/Dropbox/Thesis/rf.csv',
            index_col=0,
            header=0
        )
rf_data = rf_data / 100.0
rf_data.index = pd.to_datetime(rf_data.index)
rf_data = rf_data.sort_index()
rf_data.head()
rf_data.plot()
returns = np.log(prices/prices.shift(1))
returns = returns.dropna()
returns.head()
returns.tail()
returns.plot()
rf = np.array(rf_data.iloc[1:, 0])

# Show the original data for comparison
rf_data.head(6)

# Show the new series - is it shifted one period? Yes.
pd.DataFrame(rf[:5], columns=['RF']).head()
excess_returns = returns.sub(rf, axis=0)
excess_returns.head()
excess_returns.tail()
excess_returns.plot(linewidth=0.5, figsize=(16, 10))
cols = {1, 2}
idx = colNames

# 2013 estimates
mu = np.array([
    [0.4630, 0.2958],
    [0.2202, 0.4071],
    [0.3380, 0.1409],
    [0.7841, -0.7157],
    [0.8274, 0.0759]
])

cov = np.array([
    [
        [1.99231, 1.36487, 0.37123, 2.99855, 2.61054],
        [1.36487, 2.16692, -0.06630, 1.05048, 1.60245],
        [0.37123, -0.06630, 12.01824, 2.76506, 1.90716],
        [2.99855, 1.05048, 2.76506, 17.08546, 11.06018],
        [2.61054, 1.60245, 1.90716, 11.06018, 10.21645]
    ],
    [
        [17.54800, 3.34101, 6.55589, 22.19526, 16.02338],
        [3.34101, 4.09198, 1.55628, -0.09049, 1.46926],
        [6.55589, 1.55628, 33.26028, 8.46432, 6.03756],
        [22.19526, -0.09049, 8.46432, 72.22134, 46.50698],
        [16.02338, 1.46926, 6.03756, 46.50698, 42.42729]

    ]
])

probs = np.array([
    [0.94, 0.19],
    [0.06, 0.81]
])

print('State 1: pos. semi. def?: ', np.all(np.linalg.eigvals(cov[0]) > 0))
print('State 1: symmetric?: ', np.allclose(cov[0], cov[0].T))

print('State 2: pos. semi. def?: ', np.all(np.linalg.eigvals(cov[1]) > 0))
print('State 2 symmetric?: ', np.allclose(cov[1], cov[1].T))


"""
# 2017 estimates
mu = np.array([
    [0.3815, 0.1523],
    [0.1875, 0.3174],
    [0.0048, 0.0190],
    [0.4185, -0.4411],
    [0.6141, -0.1652]
])

cov = np.array([
    [
        [1.97569, 1.38601, 0.37623, 2.90098, 2.50570],
        [1.38601, 2.15937, -0.36313, 1.08337, 1.59527],
        [0.37623, -0.36313, 11.51764, 2.04866, 1.19757],
        [2.90098, 1.08337, 2.04866, 15.98839, 10.02691],
        [2.50570, 1.59527, 1.19757, 10.02691, 9.22943]
    ],
    [
        [15.76381, 3.35970, 5.91590, 21.88487, 15.67456],
        [3.35970, 3.23990, 1.40257, 2.59855, 2.58191],
        [5.91590, 1.40257, 31.17319, 9.71158, 7.74556],
        [21.88487, 2.59855, 9.71158, 69.81108, 47.04726],
        [15.67456, 2.58191, 7.74556, 47.04726, 41.68768]
    ]
])

probs = np.array([
    [0.95, 0.13],
    [0.05, 0.87]
])

"""

mu_df = pd.DataFrame(mu, columns=cols, index=idx)

covs = [pd.DataFrame(
    cov[i],
    index=colNames,
    columns=colNames
    ) for i in range(cov.shape[0])]

mask = np.ones_like(cov[0])
mask[np.triu_indices_from(mask)] = False  # False error call

sns.heatmap(
    mu_df.T, annot=True, cmap='RdYlBu', linewidths=2
)
plt.title('Mean excess returns')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    sns.heatmap(
        cov[i], annot=True, cmap='RdYlBu_r', mask=mask, linewidths=2, ax=ax
    )
plt.show()
print(
    'From column index to row, i.e. with probability: ',
    probs[0, 1],
    "we go from state 2 to state 1"
)
pd.DataFrame(probs,
             index=['To 1', 'To 2'],
             columns=['From 1', 'From 2'])


@jit(nopython=True)
def stateSim(S, M, start, probs, T, u, seed=12345):
    np.random.seed(seed)
    statePaths = np.ones((M, T)) * start
    state = np.ones(S)
    stateFreq = np.ones((M, S))
    for m in range(M):
        for t in range(T-1):
            # i defines state we are arriving from
            i = int(statePaths[m, t] - 1)
            for s in range(S):
                # Identifies which state we transition to
                state[s] = (
                    np.sum(probs[:s, i]) < u[m, t] <= np.sum(probs[:s+1, i])
                    )*(s+1)
            statePaths[m, t+1] = np.sum(state)
        for s in range(S):
            stateFreq[m, s] = np.sum(statePaths[m] == s + 1)
    return statePaths, stateFreq


S, M, start1, start2, T = 2, 50000, 1, 2, 60
u = np.random.random((M, T))

sp1, sf1 = stateSim(S, M, start1, probs, T, u)
sp2, sf2 = stateSim(S, M, start2, probs, T, u)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax1.plot(sp1.mean(axis=0))
ax1.set_title(
    'Starting in state 1, increasing probabilitiy of shift to state 2'
)
ax2.plot(sp2.mean(axis=0))
ax2.set_title(
    'Starting in state 2, increasing probabilitiy of shift to state 1'
)
plt.show()


@jit
def returnSim(S, M, N, A, start, mu, cov, probs, T, u, seed=12345):
    np.random.seed(seed)
    states, freq = stateSim(S, M, start, probs, T, u)
    returns = np.zeros((M*N, A, T))
    for m in range(M):
        for n in range(N):
            for s in range(S):
                returns[m*N + n, :, states[m] == s + 1] = \
                    np.random.multivariate_normal(
                        mu[:, s], cov[s], int(freq[m, s])
                    )
    return returns, states


A, N = len(colNames), 1

# import time

t0 = time.time()
ret1, s1 = returnSim(S, M, N, A, start1, mu, cov, probs, T, u)
t1 = time.time()
print('First returns computed in {}'.format(np.round(t1-t0, 4)), 'seconds.')

t0 = time.time()
ret2, s2 = returnSim(S, M, N, A, start2, mu, cov, probs, T, u)
t1 = time.time()
print('Second returns computed in {}'.format(np.round(t1-t0, 4)), 'seconds.')

ret1 /= 100
ret2 /= 100

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
fig.delaxes(ax=axes[1, 2])
for i, ax in enumerate(axes.flat[:5]):
    ax.plot(ret1[:, i, :].mean(axis=0))
    ax.set_title(colNames[i])
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
fig.delaxes(ax=axes[1, 2])
for i, ax in enumerate(axes.flat[:5]):
    ax.plot(ret2[:, i, :].mean(axis=0))
    ax.set_title(colNames[i])
plt.show()


def check_sum(weights):
    return np.sum(weights) - 1.0


def boundedOptimiser(f, w, args, ApB, method='SLSQP'):
    bnds = tuple(zip(np.zeros(ApB), np.ones(ApB)))
    cons = ({'type': 'eq', 'fun': check_sum})
    res = opt.minimize(
        f, w, args=args, bounds=bnds, constraints=cons, method=method
    )
    return res


@jit(nopython=True)
def expectedUtilityMult(w, returns, rf, g, A, T):
    rfCR = np.exp(T * rf)  # rfCR: risk free compounded return
    denominator = 1 - g
    rCR = np.exp(np.sum(returns, axis=2))*rfCR  # rCR:  risky compounded return
    numerator = (w[A] * rfCR + np.sum(w[:A] * rCR, axis=1)) ** (1 - g)
    return -np.mean(numerator / denominator) * 100000


maturities = np.array([3, 6, 9, 10, 12, 24, 36, 48, 60])
start_states = [1, 2]
gamma = [3, 5, 7, 9]

# Labelling for data frames
labels = np.hstack((colNames, 'Risk Free'))
abbrev = ['HY', 'IG', 'C', 'R2', 'R1', 'RF']

# Assets to allocate weights to
a = len(labels)  # 6

# 6 standardised random weights on the unit interval
w = np.random.random(a)
w /= np.sum(w)

# Non-simulated risk free rate of return in percent
rf = 0.003

# A matrix of weights for each asset for each maturity (3 x 6)
weights = np.squeeze(list(zip(
    [np.repeat(w[i], len(maturities)) for i in range(len(w))]
))).T

# Technicality: dictionary that can contain solutions for all states and gammas
ws = {s:
      {g: pd.DataFrame(weights.copy(), index=maturities, columns=abbrev)
       for g in gamma}
      for s in start_states}

# Technicality: Lists that contains returns of maturities 3, 6 and 12 months
R1 = [ret1[:, :, :mat] for mat in maturities]
R2 = [ret2[:, :, :mat] for mat in maturities]
R = {0: R1,
     1: R2}

# Looping; j=[0,1], s=[1,2], gamma=[3,5,7,9], i=[0,1,2], mat=[3,6,9]
for j, s in enumerate(start_states):
    print('start: ', s)
    for g in gamma:
        print('gamma: ', g)
        for i, mat in enumerate(maturities):
            args = R[j][i], rf, g, a-1, mat
            results = boundedOptimiser(expectedUtilityMult, w, args, a)
            print('weights: ', np.round(results.x, 4))
            ws[s][g].iloc[i, :] = np.round(results.x, 4)
asset_weights = {
    'aw1': {},
    'aw2': {}
}

for j, aw in enumerate(asset_weights):
    for i, a in enumerate(abbrev):
        asset_weights[aw][a] = pd.DataFrame(
            (ws[j+1][g].iloc[:, i] for g in gamma),
            index=gamma,
            columns=maturities
        ).T
    print(
        abbrev[0],
        'Start state is {}: (maturities down, gamma out)'.format(j+1))
    display(asset_weights[aw][abbrev[0]])

colors = np.array(['blue', 'green', 'red', 'black'])
fig, axes = plt.subplots(
    nrows=2, ncols=3, sharex=True, sharey=True, figsize=(15, 8)
)
for ax, a in zip(axes.flat, abbrev):
    ax.set_title(a)
    asset_weights['aw1'][a].plot(
        legend=True, ax=ax, color=colors, linewidth=0.8
    )
    asset_weights['aw2'][a].plot(
        legend=False, ax=ax, color=colors, linewidth=0.8, linestyle='dashed'
    )
plt.show()

sts = np.hstack((np.repeat(1, len(gamma)), np.repeat(2, len(gamma))))
gms = np.hstack((gamma, gamma))

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))
for s, g, ax in zip(sts, gms, axes.flat):
    ws[s][g].plot.bar(ax=ax)
    ax.set_title('Start from {}, gamma is {}'.format(s, g))
plt.show()

display(rf_data.iloc[366:378])
rf_array = np.array(rf_data.iloc[366:, 0])

returns_18 = excess_returns.iloc[365:]
returns_18 = returns_18.add(rf, axis=0)
display(returns_18[:12])

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
returns_18.plot(ax=ax1)
rf_data.iloc[366:].plot(ax=ax2)
plt.show()

pd.options.display.float_format = '{:.4f}'.format
