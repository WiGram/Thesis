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

idx = ['State 1', 'State 2', 'State 3']
states = len(idx)

labels = np.array([
    'High Yield',
    'Investment Grade',
    'Commodities',
    'Russell 2000',
    'Russell 1000'
])

llh = -4788.5056

mu = np.array([
    [0.4996, 0.7625, 0.2645],
    [0.2738, 0.2223, 0.5393],
    [-0.0434, -0.0090, -0.2123],
    [0.9257, -0.5142, 0.4201],
    [0.9598, -0.5651, 0.1754]
])

cov = np.array([
    [
        [2.0782, 1.2671, 1.0225, 3.4471, 2.9941],
        [1.2671, 1.9812, 0.0639, 0.9088, 1.4217],
        [1.0225, 0.0639, 12.8728, 3.5611, 2.6270],
        [3.4471, 0.9088, 3.5611, 18.8669, 12.2045],
        [2.9941, 1.4217, 2.6270, 12.2045, 11.0383]
    ],
    [
        [10.5456, 2.5388, -0.1986, 8.4767, 5.7667],
        [2.5388, 1.8779, 0.0348, 3.0150, 2.2837],
        [-0.1986, 0.0348, 6.6786, 11.0376, 9.3243],
        [8.4767, 3.0150, 11.0376, 36.5820, 27.8837],
        [5.7667, 2.2837, 9.3243, 27.8837, 22.7033]
    ],
    [
        [15.8621, 2.2329, 7.9319, 22.0356, 13.7921],
        [2.2329, 4.9451, -0.2129, -3.1953, -0.6483],
        [7.9319, -0.2129, 45.2411, 14.5452, 7.6370],
        [22.0356, -3.1953, 14.5452, 82.7686, 50.1592],
        [13.7921, -0.6483, 7.6370, 50.1592, 43.1839]
    ]
])

probs = np.array([
    [0.96, 0.10, 0.16],
    [0.00, 0.72, 0.15],
    [0.04, 0.18, 0.69]
])

# ============================================= #
# ===== Plotting ============================== #
# ============================================= #

sns.set_style('white')


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
        'C:/Users/willi/Dropbox/Thesis/Plots/mu_{}states.png'.format(states),
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


if __name__ == '__main__':
    plot_moments(mu, cov, labels, idx, states)
