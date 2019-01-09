# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 06:59:27 2018

@author: wigr11ab
"""

import numpy as np

def llfAr(theta, y):
    end = len(y)
    mu, rho, sd = theta
    
    mu_cond = mu + rho * y[:end-1]
    log_sd2 = np.log(2 * np.pi * sd ** 2)
    
    return 0.5 * (log_sd2 + ((y[1:] - mu_cond) / sd) ** 2)

def llfArSum(theta, y):
    return sum(llfAr(theta, y))

def llfArch(theta, y):
    end = len(y)
    sig2, alpha = np.exp(theta)

    s2 = sig2 + alpha * y[:end - 1] ** 2

    return -(-np.log(s2) - 4 * np.log(1 + y[1:] ** 2 / s2))

def llfArchSum(theta, y):
    return sum(llfArch(theta, y))

def llfGjrArch(theta, y):
    if len(theta) != 3:
        return 'Parameter must have dimension 3.'
    end = len(y)
    sig2, alpha, gamma = theta
    
    idx = (y < 0)
    s2     = sig2 + alpha * y[:end - 1] ** 2 + gamma * idx[:end - 1] * y[:end - 1] ** 2
    log_s2 = np.log(s2)

    return -0.5 * (np.log(2 * np.pi) + log_s2 + y[1:] ** 2 / s2)

def llfGjrArchSum(theta, y):
    return -sum(llfGjrArch(theta, y))

def deltaGjrArch(theta, y):
    end = len(y)
    sig2, alpha, gamma = np.exp(theta)
    yLag2 = y[:end - 1] ** 2

    s2 = sig2 + alpha * yLag2 + gamma * (y < 0)[:end - 1] * yLag2

    return -0.5 * (np.log(2 * np.pi) + np.log(s2) + y[1:] ** 2 / s2)

def deltaGjrArchSum(theta, y):
    return -sum(deltaGjrArch(theta, y))

def llfAArch(theta, y):
    end = len(y)
    sig2, alphaP, alphaN = np.exp(theta)
    
    idxP  = (y > 0)[:end - 1]
    idxN  = (y < 0)[:end - 1]
    yLag2 = y[:end - 1] ** 2

    s2 = sig2 + alphaP * idxP * yLag2 + alphaN * idxN * yLag2

    return -0.5 * (np.log(2 * np.pi) + np.log(s2) + y[1:] ** 2 / s2)

def llfAArchSum(theta, y):
    return - sum(llfAArch(theta, y))
