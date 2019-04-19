"""
This script will always contain hard-coded outputs from a
matlab script.

There is absolutely no intention of figuring out a way
for the two scripts to communicate.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

llh = -4768.6902

mu = np.array([
    [0.3785, 0.2000],
    [0.2793, 0.4405],
    [-0.0923, -0.1077],
    [0.6639, -0.3688],
    [0.9141, -0.5641]
])

ar = np.array([
    [0.1938, 0.2824],
    [0.0660, -0.0677],
    [-0.0323, 0.1546],
    [-0.0279, -0.0099],
    [-0.1354, -0.0072]
])

cov = np.array([
    [
        [1.9776, 1.1421, 0.7820, 2.9788, 2.5520],
        [1.1421, 1.9370, -0.2421, 0.7157, 1.3348],
        [0.7820, -0.2421, 11.6847, 2.7199, 1.5728],
        [2.9788, 0.7157, 2.7199, 16.9808, 10.7832],
        [2.5520, 1.3348, 1.5728, 10.7832, 9.9854]
    ],
    [
        [25.3408, 8.0008, 2.2317, 28.9156, 19.9385],
        [8.0008, 5.0525, 1.2737, 5.9491, 4.2906],
        [2.2317, 1.2737, 33.8081, 7.2009, 6.1226],
        [28.9156, 5.9491, 7.2009, 79.9800, 51.4189],
        [19.9385, 4.2906, 6.1226, 51.4189, 42.9327]
    ]
])

probs = np.array([
    [0.97, 0.12],
    [0.03, 0.88]
])

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

sns.set_style('white')


def plot_moments(mu, ar, cov, labels, idx, states):
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
            'cov{}_{}states_ar.png'.format(i+1, states),
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
        'C:/Users/willi/Dropbox/Thesis/Plots/mu_2states_ar.png',
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
        'C:/Users/willi/Dropbox/Thesis/Plots/'
        'SR_{}states_ar.png'.format(states),
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()

    ars = pd.DataFrame(ar.T, index=idx, columns=labels)
    sns.heatmap(ars, annot=True, cmap='RdYlBu', linewidth=2)
    plt.title('AR(1) coefficients')
    plt.savefig(
        'C:/Users/willi/Dropbox/Thesis/Plots/ar_2states.png',
        bbox_inches='tight',
        pad_inces=0
    )
    plt.show()


plot_moments(mu, ar, cov, labels, idx, states)
