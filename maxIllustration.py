# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:13:35 2018

@author: WiGram
"""

import likelihoodModule as llm
import plotsModule as pltm
import numpy as np
import pandas as pd
import quandl
import scipy.optimize as opt
from pandas_datareader import data as web
np.set_printoptions(suppress = True)   # Disable scientific notation

muS = np.array([-0.0510, 0.0069,  0.0116, 0.0519])
muB = np.array([-0.0131, 0.0009, -0.0023, 0.0136])

eps =  np.array([[ 0.1625, -0.4060],
                 [-0.4060,  0.0902]],
                [[0.1118, 0.2043],
                 [0.2043, 0.0688]],
                [[0.1133, 0.1521],
                 [0.1521, 0.0261]],
                [[0.1429, 0.3692],
                 [0.3692, 0.1000]])

probs = np.array([0.4940, 0.0001, 0.02409, 0.4818],
                 [0.0483, 0.8529, 0.0307,  0.0682],
                 [0.0439, 0.0701, 0.8822,  0.0038],
                 [0.0616, 0.1722, 0.0827,  0.6836])

