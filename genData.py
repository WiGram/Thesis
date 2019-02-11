"""
Date:    February 11th, 2019
Authors: Kristian Strand and William Gram
Subject: Building price and return data sets.

Description:
This script generates default Bloomberg script. It is set up to
also handle other data sets, although those would have to be
read appropriately before using as input to the genData function.
"""

import markowitzOpt as mpt
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from scipy.optimize import minimize
from matplotlib import pyplot as plt
# import matplotlib.style as style
from numba import jit
from pandas_datareader import data as web
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')
np.set_printoptions(suppress = True)   # Disable scientific notation



# Read data into a Pandas data frame

def genData(data = 'default', rfData = 'default'):

    if data == 'default':
        data = pd.read_csv(
            '/home/william/Dropbox/Thesis/mthReturns.csv', 
            index_col=0,
            header=0)
        colNames = ['High Yield','Investment Grade','Commodities','Russell 2000','Russell 1000','S&P 500']
    else:
        # Extract column names
        colNames = list(data)
    
    if rfData == 'default':
        rfData = pd.read_csv(
            '/home/william/Dropbox/Thesis/rf.csv',
            index_col=0,
            header=0
        )

    # Set format of index to a date format
    data.index = pd.to_datetime(data.index)
    rfData.index = pd.to_datetime(rfData.index)

    # Sort data with oldest data first
    data = data.sort_index()
    rfData = rfData.sort_index()

    # Count amount of assets
    assets = len(colNames)

    # Return index dates
    pDates = data.index

    # ============================================= #
    # ===== Analysis of returns =================== #
    # ============================================= #

    rf = np.array(rfData.iloc[1:,0]) / 100

    # Applying actual returns convention
    monthlyRets = data/data.shift()-1
    monthlyRets = monthlyRets.iloc[1:,:]

    rDates = monthlyRets.index

    # Hard coded return and volatility column names
    monthlyRets.columns = colNames

    # Var is squared return process, assuming true mean return of 0
    monthlyVol = np.sqrt(monthlyRets ** 2)

    # monthlyRets.cov() * 12 # Yearly covariance matrix
    retCov = monthlyRets.cov()

    # Subtract risk free rate for excess monthly returns
    excessMRets = monthlyRets.sub(rf, axis = 'rows')
    
    return data, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates
