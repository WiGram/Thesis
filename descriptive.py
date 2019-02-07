"""
Date:    February 6th, 2019
Authors: Kristian Strand and William Gram
Subject: Describing Bloomberg time series

Description:
This script is strictly exploratory, considering
monthly Bloomberg data. Time series are illustrated
and the first two moments are estimated.
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from matplotlib import pyplot as plt
# import matplotlib.style as style
from numba import jit
from pandas_datareader import data as web
# import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-paper')
np.set_printoptions(suppress = True)   # Disable scientific notation

bbData = pd.read_csv('/home/william/Dropbox/Thesis/mthReturns.csv', index_col=0,header=0)
bbData.index = pd.to_datetime(bbData.index, format = '%Y-%m-%d')
bbData = bbData.sort_index()
colNames = list(bbData)
rows = len(bbData[colNames[0]])
d = bbData.index

bbData.iloc[:,:len(colNames)].plot()
plt.show()

retList = ['HY_ret','IG_ret','CD_ret','R2_ret','R1_ret','SP_ret']

for i in range(len(retList)):
    bbData[retList[i]] = bbData[colNames[i]].pct_change()

# 321: Plot rows = 3, plot columns = 2, plot is no. 1
plotPos = [321, 322, 323, 324, 325, 326]

for i in range(len(plotPos)):
    plt.subplot(plotPos[i])
    plt.plot(d, bbData[colNames[i]])
plt.subplots_adjust(top = 0.92, bottom = 0.08, left = 0.10, right = 0.95,
                    hspace = 0.5, wspace = 0.35)
plt.show()



for i in range(len(plotPos)):
    plt.subplot(plotPos[i])
    plt.plot(d, bbData[retList[i]])
plt.show()

descriptive = bbData.describe()

# With pandas: use iloc for slicing by indices
retDes = descriptive.iloc[:,len(colNames):]

vols  = retDes.iloc[2,:]
means = retDes.iloc[1,:]
SR = means / vols

SR.plot.bar(y = SR)
plt.bar(SR, 1)
plt.show()

sns.pairplot(bbData.iloc[:,len(colNames):])