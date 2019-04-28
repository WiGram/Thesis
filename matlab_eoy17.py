"""
This script will always contain hard-coded outputs from a
matlab script.

There is absolutely no intention of figuring out a way
for the two scripts to communicate.

Specific to this script
-----------------------
These data are based on a subset of the entire data series.
We have subsetted the data to end in december 2017, with one year of
data remaining.

This construction is made such that we may simulate 1 year of data based
on historical data up to and including december 2017. Then, we want to
generate portfolios with investment horizons of 3, 6 and 12 months and compare
with optimal portfolios given the actual 12 month data.
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

llh = -4804.6401


"""
mu = np.array([
    [0.4688, -0.1432],
    [0.2543, -0.3492],
    [-0.0033, 0.1711],
    [0.5783, -0.2671],
    [0.7502, -0.1591]
])

"""
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
    [0.94, 0.16],
    [0.06, 0.84]
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
