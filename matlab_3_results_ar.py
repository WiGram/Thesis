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

llh = -4735.3998

mu = np.array([
    [0.4105, 0.0710, 0.2238],
    [0.3211, 0.1296, 0.3511],
    [0.1495, 0.1093, -0.0626],
    [0.6855, -0.2514, -0.0876],
    [1.0916, -0.0507, 0.2591]
])

ar = np.array([
    [0.3169, 0.0502, 0.3623],
    [0.1868, 0.0409, -0.0347],
    [0.2232, -0.1243, 0.1142],
    [0.1156, -0.0560, 0.1093],
    [0.0031, -0.1418, 0.1261]
])

cov = np.array([
    [
        [1.2006, 1.1000, 0.7506, 2.3962, 2.3173],
        [1.1000, 1.9355, 0.9789, 1.6574, 2.2567],
        [0.7506, 0.9789, 8.6826, 2.2219, 1.5525],
        [2.3962, 1.6574, 2.2219, 16.5714, 10.3988],
        [2.3173, 2.2567, 1.5525, 10.3988, 9.8000]
    ],
    [
        [4.0074, 1.6841, 2.3193, 6.2836, 5.1106],
        [1.6841, 2.3189, -0.1558, 0.7953, 1.1520],
        [2.3193, -0.1558, 19.0090, 7.2474, 5.6171],
        [6.2836, 0.7953, 7.2474, 25.2124, 17.4227],
        [5.1106, 1.1520, 5.6171, 17.4227, 14.2680]
    ],
    [
        [14.7041, 4.7736, 3.4551, 19.8027, 14.7796],
        [4.7736, 4.0691, 2.4736, 4.2193, 4.0381],
        [3.4551, 2.4736, 28.8396, 4.8848, 4.8021],
        [19.8027, 4.2193, 4.8848, 67.7417, 45.1555],
        [14.7796, 4.0381, 4.8021, 45.1555, 42.6938]
    ]
])

probs = np.array([
    [0.96, 0.03, 0.02],
    [0.00, 0.93, 0.19],
    [0.04, 0.05, 0.79]
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
        'C:/Users/willi/Dropbox/Thesis/Plots/'
        'mu_{}states_ar.png'.format(states),
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
        'C:/Users/willi/Dropbox/Thesis/Plots/'
        'ar_{}states.png'.format(states),
        bbox_inches='tight',
        pad_inces=0
    )
    plt.show()


if __name__ == '__main__':
    plot_moments(mu, ar, cov, labels, idx, states)
