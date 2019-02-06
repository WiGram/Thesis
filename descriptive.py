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
from numba import jit
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

bbData = pd.read_csv('/home/william/Dropbox/Thesis/mthReturns.csv', index_col=0,header=0)
d = pd.to_datetime(bbData.index)

highYield = bbData['LF98TRUU']
investGrd = bbData['LUACTRUU']
commodTR  = bbData['BCOMTR']
russell2k = bbData['Russell2000']
russell1k = bbData['Russell1000']
sp500TR   = bbData['SPXT']

pltm.plotUno(d, ts)

""" 
Check out following site for plotting: 
https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
"""

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.add_subplot()
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')

# 321 => 3 plots vertically, 2 plots horizontally, figure no. 1
plt.subplot(321)
plt.plot(d, highYield)
plt.subplot(322)
plt.plot(d, investGrd)
plt.subplot(323)
plt.plot(d, commodTR)
plt.subplot(324)
plt.plot(d, russell2k)
plt.subplot(325)
plt.plot(d, russell1k)
plt.subplot(326)
plt.plot(d, sp500TR)
plt.show()