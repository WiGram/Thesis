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

import matplotlib.pyplot as plt
import numpy as np
import simulateSimsReturns as ssr
import seaborn as sns
sns.set_style('white')

# ============================================= #
# ===== Two states ============================ #
# ============================================= #


def return_attributes():
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

    att_keys = ['mu', 'ar', 'var', 'cov', 'probs']
    att_vals = [M, AR, V, C, P]

    attributes = dict(zip(att_keys, att_vals))
    return attributes


if __name__ == '__main__':

    atts = return_attributes()
    M, AR, V, C, P = atts.values()

    N = 1  # simulations per simulated path
    keys = list(M.keys())
    S = len(list(M.values())[0])  # amount of states
    T = 120  # Amount of periods
    SPs = 1000  # StatePath simulations
    start = 1
    u = np.random.uniform(0, 1, (SPs, T))

    paths = {}
    k = keys[0]
    for k in keys:
        paths[k], _ = ssr.stateSim(S, SPs, start, P[k], T, u, seed=12345)

    test = ssr.returnSimUniMix(
        S, SPs, N, start, V, M, AR, C, P, T, paths
    )

    plt.plot(test[:, 4, :].mean(axis=0))
    plt.show()
