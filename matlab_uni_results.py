"""
Parameter attributes
-------------------------------------------------
llh = [HY, IG, C, R2, R1]
mu = [[HY_1, HY_2],[IG_1,IG_2],...,[R1_1,R1_2]]
var = [[HY_1, HY_2],[IG_1,IG_2],...,[R1_1,R1_2]]
probs = [
    [HY_1->1, HY_2->1],
    [HY_1->2, HY_2->2],
    ...
    [R1_1->1, R1_2->1],
    [R1_1->2, R1_2->2]
]
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simulateSimsReturns as ssr
sns.set_style('white')

# ============================================= #
# ===== Two states ============================ #
# ============================================= #

idx = ['State 1', 'State 2']
states = len(idx)
labels = np.array([
    'High Yield',
    'Investment Grade',
    'Commodities',
    'Russell 2000',
    'Russell 1000'
])
keys = ['HY', 'IG', 'C', 'R2', 'R1']

mu = np.array([
    [0.4802, -0.0163],
    [0.2585, 0.4501],
    [0.3221, -0.2060],
    [1.0764, -3.4049],
    [1.2517, -0.7699]
])
M = dict(zip(keys, mu))

ar = np.array([
    [0.1745, 0.3733],
    [0.1001, 0.2888],
    [0.0377, 0.0893],
    [-0.0240, 0.1372],
    [-0.1264, 0.1108]
])
AR = dict(zip(keys, ar))

var = np.array([
    [1.3241, 12.8401],
    [1.6213, 7.8316],
    [4.3768, 22.4948],
    [18.2258, 89.2267],
    [8.2150, 36.5953]
])
V = dict(zip(keys, var))

np.any(np.linalg.eigvals(C)< 0)
eigen_values, eigen_vector = np.linalg.eig(C)

eigen_values[eigen_values < 0] = 0.0001
D = np.diag(eigen_values)
C_PD = np.round(evec @ D @ evec.T, 4)

C
C_PD
C_PD - C


C = np.array([
    [5.6666, 1.9944, 2.2761, 8.0505, 6.1767],
    [1.9944, 2.5016, 0.5334, 1.5312, 1.8945],
    [2.2761, 0.5334, 17.0958, 4.8942, 3.6903],
    [8.0505, 1.5312, 4.8942, 31.2047, 20.8037],
    [6.1767, 1.8945, 3.6903, 20.8037, 18.6928]
])

probs = np.array([
    [[0.95, 0.11],
     [0.05, 0.89]],
    [[0.99, 0.08],
     [0.01, 0.92]],
    [[0.96, 0.02],
     [0.04, 0.98]],
    [[0.95, 0.28],
     [0.05, 0.72]],
    [[0.95, 0.09],
     [0.05, 0.91]]
])
P = dict(zip(keys, probs))
S = states
T = 120
SPs = 50000  # StatePath simulations
start = 1
u = np.random.uniform(0, 1, (SPs, T))

state_paths = {}
for k in keys:
    state_paths[k], _ = ssr.stateSim(S, SPs, start, P[k], T, u, seed=12345)

N = 1  # simulations per simulated path
k = 'HY'

plt.plot(state_paths[k][0, :])

test = ssr.returnSimUniMix(S, 1000, N, start, V, M, AR, C, P, T, state_paths)

m = 0
k = keys[0]
n = 0
t = 0
SP = sims

def returnSimUniMix(S, SP, N, start, V, M, A, C, P, T, paths, seed=12345):
    """
    Produces
    ---------------------------------
    Simulates M*N matrices (AxT) of return processes for each
    asset for a lenght of time T;
    Return processes for risky assets are generated
    by multivariate normal distribution.
    Each asset has its own state path.
    The covariances are static (variances switch)

    Inputs
    ---------------------------------
    S:     Scalar indicating amount of states
    SP:    Scalar indicating amount of state path simulations
    N:     Scalar indicating amount of return simulations
    start: Scalar indicating initial state to simulate from (state, not index)
    V:     (A x S) State switching variances
    M:     (A x S) matrix of returns for each asset in each state
    A:     (A x S) matrix of autoregressive coefs for each asset and state
    C:     (A x A) State invariant (i.e. static) covariance matrix
    P:     (A x (S x S)) Transition probability matrices for each asset
    T:     Scalar simulation length, e.g. T = 12 months
    paths: (A x (SP x T)) SP amount of simulated state paths for each asset.

    Returns
    ---------------------------------
    returns:  (M*N x A x T) M*N simulated returns of length T for A assets
    """
    keys = M.keys()
    np.random.seed(seed)
    returns = np.zeros((SP*N, A, T))
    for m in range(SP):
        for n in range(N):
            for t in range(T-1):
                var = [V[k][paths[k][m*N+n, t]-1] for k in keys]
                cov[np.diag_indices_from(cov)] = var
                ar = [A[k][paths[k][m*N+n, t]-1] for k in keys]
                mu = [M[k][paths[k][m*N+n, t]-1] for k in keys]
                mean = mu + ar * returns[m*N+n, :, t]
                returns[m*N + n, :, t+1] = \
                    np.random.multivariate_normal(mean, cov)
    return returns
# ============================================= #
# ===== Plotting ============================== #
# ============================================= #


def plot_properties(mu, cov, labels, idx, states):



def plot_moments(mu, cov, labels, idx, states):
    cov = [pd.DataFrame(
        cov[i],
        index=labels,
        columns=labels
    ) for i in range(states)]

    mask = np.ones_like(cov[0])
    mask[np.triu_indices_from(mask)] = False  # False error call

    for i in range(states):
        sns.heatmap(
            cov[i], annot=True, cmap='RdYlBu_r', mask=mask, linewidths=2
        )
        plt.title('Covariance matrix, starting in state {}'.format(i+1))
        plt.savefig(
            'C:/Users/willi/Dropbox/Thesis/Plots/'
            'cov{}_{}states.png'.format(i+1, states),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.show()

    # ===== Mean returns ========================== #
    mus = pd.DataFrame(mu.T, index=idx, columns=labels)

    sns.heatmap(
        mus, annot=True, cmap='RdYlBu', linewidths=2
    )
    plt.title('Mean excess returns')
    plt.savefig(
        'C:/Users/willi/Dropbox/Thesis/Plots/mu_2states.png',
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()

    # ===== SR ==================================== #
    SR = pd.DataFrame(
        [mu[:, i].T/np.sqrt(np.diag(cov[i])) for i in range(states)],
        index=idx,
        columns=labels
    )

    sns.heatmap(
        SR, annot=True, cmap='RdYlBu', linewidths=2
    )
    plt.title('Sharpe Ratios')
    plt.savefig(
        'C:/Users/willi/Dropbox/Thesis/Plots/SR_{}states.png'.format(states),
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()
