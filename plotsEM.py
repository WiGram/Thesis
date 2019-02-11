# -*- coding: utf-8 -*-
"""
Date:    February 11th, 2019
Authors: Kristian Strand and William Gram
Subject: Plots output from the EM algorithm

Description:
This script handles plotting of time series from the EM
algorithm.
"""

import plotsModule as pltm
import numpy as np
import pandas as pd
np.set_printoptions(suppress = True)   # Disable scientific notation

def returnPlots(states, sims, vs, ms, ps, pStar, llh, colNames, d):
    if states == 2:
        pltm.plotDuo(range(sims), vs[0,:], vs[1,:], 'Var_1', 'Var_2', 'Trials', 'Variance')
        pltm.plotDuo(range(sims), ms[0,:], ms[1,:], 'Mu_1', 'Mu_2', 'Trials', 'Mean return')
        pltm.plotDuo(range(sims), ps[0,:], ps[3,:], 'p11', 'p22', 'Trials', 'Probability')
        pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')
    elif states == 3:
        # vs: vs[m, s, asset, asset] -> covariance for vs[m, s, 0, 1] or vs[m, s, 1, 0]
        pltm.plotTri(range(sims), vs[:,0,0,0], vs[:, 1, 0, 0], vs[:, 2, 0, 0], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % colNames[0]))
        pltm.plotTri(range(sims), vs[:,0,1,1], vs[:, 1, 1, 1], vs[:, 2, 1, 1], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % colNames[1]))
        pltm.plotTri(range(sims), vs[:,0,2,2], vs[:, 1, 2, 2], vs[:, 2, 2, 2], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % colNames[2]))
        pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 3, 3], vs[:, 2, 3, 3], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % colNames[3]))
        pltm.plotTri(range(sims), vs[:,0,3,3], vs[:, 1, 4, 4], vs[:, 2, 4, 4], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Variance', title = ('Variance of %s' % colNames[4]))
        pltm.plotTri(range(sims), ms[:, 0, 0], ms[:, 0, 1], ms[:, 0, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % colNames[0]))
        pltm.plotTri(range(sims), ms[:, 1, 0], ms[:, 1, 1], ms[:, 1, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % colNames[1]))
        pltm.plotTri(range(sims), ms[:, 2, 0], ms[:, 2, 1], ms[:, 2, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % colNames[2]))
        pltm.plotTri(range(sims), ms[:, 3, 0], ms[:, 3, 1], ms[:, 3, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % colNames[3]))
        pltm.plotTri(range(sims), ms[:, 4, 0], ms[:, 4, 1], ms[:, 4, 2], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mean return', title = ('Mean of %s' % colNames[4]))
        # pltm.plotTri(range(sims), arS[:, 0, 0], arS[:, 0, 1], arS[:, 0, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % colNames[0]))
        # pltm.plotTri(range(sims), arS[:, 1, 0], arS[:, 1, 1], arS[:, 1, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % colNames[1]))
        # pltm.plotTri(range(sims), arS[:, 2, 0], arS[:, 2, 1], arS[:, 2, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % colNames[2]))
        # pltm.plotTri(range(sims), arS[:, 3, 0], arS[:, 3, 1], arS[:, 3, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % colNames[3]))
        # pltm.plotTri(range(sims), arS[:, 4, 0], arS[:, 4, 1], arS[:, 4, 2], 'Trials', 'AR_1', 'AR_2', 'AR_3', 'AR coefficient', title = ('AR of %s' % colNames[4]))
        # pltm.plotTri(range(sims), avgS[:, 0, 0], avgS[:, 0, 1], avgS[:, 0, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % colNames[0]))
        # pltm.plotTri(range(sims), avgS[:, 1, 0], avgS[:, 1, 1], avgS[:, 1, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % colNames[1]))
        # pltm.plotTri(range(sims), avgS[:, 2, 0], avgS[:, 2, 1], avgS[:, 2, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % colNames[2]))
        # pltm.plotTri(range(sims), avgS[:, 3, 0], avgS[:, 3, 1], avgS[:, 3, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % colNames[3]))
        # pltm.plotTri(range(sims), avgS[:, 4, 0], avgS[:, 4, 1], avgS[:, 4, 2], 'Trials', 'Avg_1', 'Avg_2', 'Avg_3', 'Average', title = ('Avg of %s' % colNames[4]))
        pltm.plotTri(range(sims), ps[:,0,0], ps[:, 1, 1], ps[:, 2, 2], 'Trials', 'p11', 'p22', 'p33', 'Probability')
        pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
        pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
        pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
        pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')
    elif states == 4:
        pltm.plotQuad(range(sims), vs[0,:], vs[1,:], vs[2,:], vs[3,:], 'Trials', 'Var_1', 'Var_2', 'Var_3', 'Var_4', 'Variance')
        pltm.plotQuad(range(sims), ms[0,:], ms[1,:], ms[2,:], ms[3,:], 'Trials', 'Mu_1', 'Mu_2', 'Mu_3', 'Mu_4', 'Mean return')
        pltm.plotQuad(range(sims), ps[0,:], ps[5,:], ps[10,:], ps[15,:], 'Trials', 'p11', 'p22', 'p33', 'p44', 'Probability')
        pltm.plotUno(d, pStar[0,:], xLab = 'Time', yLab = 'p1', title = 'Smoothed State Probabilities')
        pltm.plotUno(d, pStar[1,:], xLab = 'Time', yLab = 'p2', title = 'Smoothed State Probabilities')
        pltm.plotUno(d, pStar[2,:], xLab = 'Time', yLab = 'p3', title = 'Smoothed State Probabilities')
        pltm.plotUno(d, pStar[3,:], xLab = 'Time', yLab = 'p4', title = 'Smoothed State Probabilities')
        pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')