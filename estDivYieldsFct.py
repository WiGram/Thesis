import numpy as np
import pandas as pd
from scipy import optimize as opt
import statsmodels.api as sm
np.set_printoptions(suppress = True)   # Disable scientific notation

def testStat(lStd, lTest):
    return - 2 * (lStd - lTest)

def pValue(testStat, type = 'normal'):
    if type == 'normal':
        return ValueError('WIP')

def densityFct(z, mean, vol):
    return 1/np.sqrt(2.0*np.pi*vol**2)*np.exp(-0.5*(z-mean)**2/vol**2)

def modelStd(z):
    return - np.sum(  np.log(densityFct(z, 0.0, 1.0)))

def modelNorm(params, z):
    alpha = params[0]  # alpha
    gamma  = params[1]  # sigma
    #
    mean = alpha
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z, mean, vol))  )*10000

def modelARone(params, z):
    z_lag  = z[1:len(z)-2]
    z_lead = z[2:len(z)-1]
    #
    alpha = params[0]
    gamma = params[1] # sigma = exp(gamma) <= ensures sigma > 0
    arOne = params[2]
    # 
    mean = alpha + arOne * z_lag
    vol  = np.exp(gamma)
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwo(params, z):
    z_llag = z.shift(2)[2:]
    z_lag  = z.shift(1)[2:]
    z_lead = z[2:]
    # 
    alpha = params[0]
    gamma = params[1]
    arOne = params[2]
    arTwo = params[3]
    #
    mean = alpha + arOne*z_lag + arTwo*z_llag
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log(densityFct(z_lead, mean, vol))  )

def modelARtwoS(params, z):
    z_llag = z.shift(2)[2:]
    z_lag  = z.shift(1)[2:]
    z_lead = z[2:]
    # 
    alpha  = params[0]
    gamma  = params[1]
    arOne  = params[2]
    arTwo  = params[3]
    arOneS = params[4]
    arTwoS = params[5]
    #
    mean = alpha+arOne*z_lag+arTwo*z_llag+arOneS*(z_lag**2)+arTwoS*(z_llag**2)
    vol  = np.exp(gamma)
    #
    return - np.sum(  np.log( densityFct(z_lead, mean, vol) )  )

# Augmented Dickey Fuller test: H0 there is a unit root in the
# AR(p) process. Specifically, we have set p=3
def DF_statistic(pi_stat):
    if pi_stat < -2.56:
        print('Reject H0 at 1%: {}***'.format(round(pi_stat,4)))
    elif pi_stat < -1.94:
        print('Reject H0 at 5%: {}**'.format(round(pi_stat,4)))
    elif pi_stat < -1.62:
        print('Reject H0 at 10%: {}*'.format(round(pi_stat,4)))
    else:
        print('H0 is not rejected, value: {}'.format(round(pi_stat,4)))

def DF_const(pi_stat):
    if pi_stat < -3.43:
        print('Reject H0 at 1%: {}***'.format(round(pi_stat,4)))
    elif pi_stat < -2.86:
        print('Reject H0 at 5%: {}**'.format(round(pi_stat,4)))
    elif pi_stat < -2.57:
        print('Reject H0 at 10%: {}*'.format(round(pi_stat,4)))
    else:
        print('H0 is not rejected, value: {}'.format(round(pi_stat,4)))

def DF_const_LR(pi_stat):
    if pi_stat > 12.73:
        print('Reject H0 at 1%: {}***'.format(round(pi_stat,4)))
    elif pi_stat > 9.13:
        print('Reject H0 at 5%: {}**'.format(round(pi_stat,4)))
    elif pi_stat > 7.50:
        print('Reject H0 at 10%: {}*'.format(round(pi_stat,4)))
    else:
        print('H0 is not rejected, value: {}'.format(round(pi_stat,4)))

