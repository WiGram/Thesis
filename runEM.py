# -*- coding: utf-8 -*-
"""
Date:    February 11th, 2019
Authors: Kristian Strand and William Gram
Subject: Describing Bloomberg time series

Description:
This script runs the EM algorithm by calling the EM_NM_EX.py script
by generating data in the genData.py script. Finally the output is
plotted by calling the plotsEM.py script.
"""

import genData as gd
import EM_NM_EX as em
import plotsEM as pem
import numpy as np
import pandas as pd

prices, monthlyRets, excessMRets, colNames, assets, monthlyVol, retCov, rf, pDates, rDates = gd.genData()

returns = np.array(monthlyRets.T)

sims   = 100
states = 3
mat    = len(returns[0,:])
p      = np.repeat(1.0 / states, states * states)
pS     = np.random.uniform(size = states * mat).reshape(states, mat)

ms, vs, ps, llh, pStar, pStarT = em.EM(returns, sims, mat, states, assets, p, pS)
pem.returnPlots(states, sims, vs, ms, ps, pStar, llh, colNames, rDates)