import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize as opt
import statsmodels.api as sm
from estDivYieldsFct import *
np.set_printoptions(suppress = True)   # Disable scientific notation

# Set matplotlib style
import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')

# ============================================= #
# ===== Graphical analysis ==================== #
# ============================================= #

# Run pfClassWIP.py until and including pf=portfolio()
from pfClassWIP import portfolio
prices=pf.prices.iloc[1:,:].copy() # copy makes prices an individual object
prices['Exogenous']=pf.exogenous
prices.iloc[:,:5].plot()
prices.iloc[:,5].plot(secondary_y=True)
plt.show()
# The plot tends to a negative correlation between div yield and other assets

prices['Exogenous'].min()
prices['Exogenous'].argmin() # deprecated, but works for now
prices['Exogenous'].idxmin() # equivalent, works in future

mini=np.array(prices['Exogenous']).argmin()
ex_pre=prices.iloc[:mini+1,:].corr()['Exogenous']
ex_post=prices.iloc[mini:,:].corr()['Exogenous']
ex_tot=prices.corr()['Exogenous']
correlations=ex_tot.to_frame(
    name='Total period'
).join(ex_pre.to_frame(
    name='Pre IT bubble'
).join(ex_post.to_frame(
    name='Post IT bubble'
)))
pd.concat([ex_tot,ex_pre,ex_post],axis=1)
# Formally also, there is a negative correlation

# Next step: model dividend yield
prices.iloc[:,5].plot(title='Development in dividend yield')
plt.xlabel('')
plt.tight_layout()
plt.show()

prices['FD Div. Yield']=prices['Exogenous'].diff()
prices['Sq. FD Div. Yield']=prices['FD Div. Yield']**2
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,4))
prices.iloc[1:,6].plot(ax=axes[0])
plt.xlabel('')
prices.iloc[:,7].plot(ax=axes[1])
axes[0].set_xlabel('')
axes[0].set_title('First difference of dividend yield')
axes[1].set_xlabel('')
axes[1].set_title('Squared first difference of dividend yield')
#fig.suptitle('bob')
plt.tight_layout()
plt.show()

# The first two moments
prices['FD Div. Yield'].mean()
prices['FD Div. Yield'].var()

# ============================================= #
# ===== Formal testing: Unit Roots ============ #
# ============================================= #

"""
z = prices
statistics = pd.DataFrame(
    np.zeros((3,5)),
    columns=['Standard','Normal','AR(1)','AR(2)','Ext. AR(2)'],
    index=['Log likelihood value','t-stat','p_val']
)

parNorm   = np.array([-0.04, 1.0])
parARone  = np.array([0.04, 1.0, 0.1])
parARtwoS = np.array([0.04, 1.0, 0.05, 0.04, 0.03, 0.02])

parameters = pd.DataFrame(
    np.zeros((5,6)),
    columns=['Mean','Volatility','AR(1)','AR(2)','Sq. AR(1)','Sq. AR(2)'],
    index=['Standard','Normal','AR(1)','AR(2)','AR(2) with squares']
)

parameters['Volatility'] = np.ones(5)

statistics.iloc[0,0]=-modelStd(z[2:])
resN=opt.minimize(modelNorm,parNorm,args=z[2:])
resARone=opt.minimize(modelARone,parARone,args=z[1:])
plt.plot(z)
plt.show()
"""
z=np.array(prices['Exogenous'].copy())
# E: Exogenous, C: Constant, X: First lag, Y: Second, Z: Third
data = {
    'div(t)':z[3:],
    'const':np.ones(len(z[3:])),
    'div(t-1)':z[2:len(z)-1],
    'div(t-2)':z[1:len(z)-2],
    'div(t-3)':z[:len(z)-3]
}

modelData = pd.DataFrame(data)

result = sm.OLS(
    endog=modelData['div(t)'],
    exog=modelData[['const','div(t-1)','div(t-2)','div(t-3)']]
).fit()
print(result.summary())

"""
Model: X = z_t - z_(t-1) = pi _(t-1) + eps_t
H(0): pi=0
H(A): -2 < pi < 0

Intuition:
1. If pi=0, then z_t = z_(t-1) + eps_t ~ random walk
2. If pi=-2, then z_t = -z_(t-1) + eps_t ~ random walk
3. If pi not in (-2,0) then explosive process
"""

modelData['d_div(t)']=modelData['div(t)']-modelData['div(t-1)']
modelData['d_div(t-1)']=modelData['div(t-1)'] - modelData['div(t-2)']
modelData['d_div(t-2)']=modelData['div(t-2)'] - modelData['div(t-3)']

result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData[['const','div(t-1)','d_div(t-1)','d_div(t-2)']]
).fit()
result.summary()

# d_div's are normally distributed. If they are insignificant -> simplify model
result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData[['const','div(t-1)','d_div(t-1)']]
).fit()
print(result.summary())

# All d_div(lags) appeared to be insignificant,
# We do not remove all however, as we later want
# to test for no constant, no unit root, hence:
pi_stat=result.params[1]/result.bse[1]
_=DF_statistic(pi_stat)

# We cannot reject H0 of a unit root. Had we removed
# the insignificant first difference:
result_if=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData[['const','div(t-1)']]
).fit()
print(result_if.summary())
pi_stat_if=result_if.params[1]/result_if.bse[1]
_=DF_const(pi_stat_if)

# We would reject a unit root in dividend yield on a 10 pct. level.
# In effect, this should be treated like a unit root.

# Test for a constant and a unit root:
result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData[['const','div(t-1)','d_div(t-1)']]
).fit()
L_A=result.llf

result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData[['d_div(t-1)']]
).fit()
L_0=result.llf
LR=-2*(L_0-L_A)
DF_const_LR(LR)

# We conclude, that the H0 cannot be rejected, that is, we cannot reject
# that dividend yield is a unit root process with a constant

# In conclusion, to model the dividend yield moving forward, we need
# to model the first difference without a constant term. Let's
# have a look at the first difference process:

modelData['d_div(t)'].plot()
plt.show()

# The plot does in fact seem to be stationary around zero,
# albeit with heteroskedastic variance

plt.plot(modelData['d_div(t)']**2)
plt.show()

# The above plot does show the serial dependance, i.e. non-stationarity
# with regards to variance.

# We now want to test for ARCH effects. Do a linear regression of div_yield
# on its first-order lag, and save the residuals
result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData['const']
).fit()
print(result.summary())
resids = np.array(result.resid)

res_data = {
    'const':np.ones(len(resids[5:])),
    'eps2(t)':resids[5:]**2,
    'eps2(t-1)':resids[4:len(resids)-1]**2,
    'eps2(t-2)':resids[3:len(resids)-2]**2,
    'eps2(t-3)':resids[2:len(resids)-3]**2,
    'eps2(t-4)':resids[1:len(resids)-4]**2,
    'eps2(t-5)':resids[0:len(resids)-5]**2
}

modelResids = pd.DataFrame(res_data)

resid_results=sm.OLS(
    endog=modelResids['eps2(t)'],
    exog=modelResids.loc[:, modelResids.columns != 'eps2(t)']
).fit()
print(resid_results.summary())
print(resid_results.summary().as_latex())
# H0: no ARCH effects, which we expect to reject at a high level
r2=resid_results.rsquared

# Test stat is T*R^2 ~ Chi^2
len(resids)*r2

# The result is ~24.1 which is highly significant at any comfortable
# confidence level.

# In conclusion, there is strong evidence of ARCH(1) effects in the
# first difference of dividend yields.

result=sm.OLS(
    endog=modelData['d_div(t)'],
    exog=modelData['const']
).fit()

print(result.summary().as_latex())

def sig2_fct(theta,y):
    o,a=np.exp(theta)
    sig2=o+a*y[:len(y)-1]**2
    return sig2

def arch_llh(theta,y):
    sig2=sig2_fct(theta,y)
    llh=-0.5*(np.log(2.*np.pi)+np.log(sig2)+y[1:]**2/sig2)
    return -np.sum(llh)

theta=np.array([0.5,0.8])
y=np.array(modelData['d_div(t)'])
arch_llh(theta,y)

res=opt.minimize(arch_llh,theta,args=y)
omega,alpha=np.exp(res.x)

eps=y[1:]/np.sqrt(omega+alpha*y[:len(y)-1]**2)

import statsmodels.api as sm
sm.qqplot(eps, line='45')
plt.show()

def arch2_llh(theta,y):
    omega=np.exp(theta[0])
    alpha1=np.exp(theta[1])
    alpha2=np.exp(theta[2])
    
    mat=len(y)
    
    lead=y[2:]
    lag1=y[1:mat-1]**2
    lag2=y[:mat-2]**2
    
    sig2=omega+alpha1*lag1+alpha2*lag2
    
    llh=-0.5*(np.log(2*np.pi)+np.log(sig2)+lead**2/sig2)
    
    return-np.sum(llh)

theta=np.array([0.5,0.6,0.3])

res=opt.minimize(arch2_llh,theta,y)
omega,alpha1,alpha2=np.exp(res.x)
sig2=omega+alpha1*y[1:len(y)-1]**2+alpha2*y[:len(y)-2]**2
eps2=y[2:]/np.sqrt(sig2)

sm.qqplot(eps2,line='45')
plt.show()

def arch5_llh(theta,y):
    omega,a1,a2,a3,a4,a5=np.exp(theta)
    mat=len(y)
    lead=y[5:]**2
    lag1=y[4:mat-1]**2
    lag2=y[3:mat-2]**2
    lag3=y[2:mat-3]**2
    lag4=y[1:mat-4]**2
    lag5=y[:mat-5]**2
    
    sig2=omega+a1*lag1+a2*lag2+a3*lag3+a4*lag4+a5*lag5
    
    llh=-.5*(np.log(2*np.pi)+np.log(sig2)+lead/sig2)
    return-np.sum(llh)

theta=np.random.uniform(size=6)
res=opt.minimize(arch5_llh,theta,y)
o,a1,a2,a3,a4,a5=np.exp(res.x)
mat=len(y)
lead=y[5:]
lag1=y[4:mat-1]**2
lag2=y[3:mat-2]**2
lag3=y[2:mat-3]**2
lag4=y[1:mat-4]**2
lag5=y[:mat-5]**2
sig2=o+a1*lag1+a2*lag2+a3*lag3+a4*lag4+a5*lag5
eps5=lead/np.sqrt(sig2)
sm.qqplot(eps5,line='45')
plt.show()

# Regardless of the amount of lags, I seem to be getting fat tails
# Enter GARCH model
theta=np.random.uniform(size=3)

from numba import jit

def gen_garch_sig(theta,y):
    o,a,b=np.exp(theta)
    sig2=np.zeros(len(y))
    for t in range(len(y)-1):
        sig2[t+1]=o+a*y[t]**2+b*sig2[t]
    return sig2

def garch_llh(theta,y):
    sig2=gen_garch_sig(theta,y)
    llh=-.5*(np.log(2*np.pi)+np.log(sig2[1:])+y[1:]**2/sig2[1:])
    return-np.sum(llh)

garch_llh(theta,y)
res=opt.minimize(garch_llh,theta,y)

theta=np.exp(res.x)
sig2=gen_garch_sig(theta,y)
eps=y[1:]/np.sqrt(sig2[1:])
sm.qqplot(eps,line='45')
plt.show()

def sim_garch(theta,y):
    m=len(y)
    o,a,b=theta
    eps=np.random.normal(size=m)
    sig2=np.zeros(m)
    x=np.zeros(m)
    for t in range(m-1):
        sig2[t+1]=o+a*x[t]**2+b*sig2[t]
        x[t+1]=np.sqrt(sig2[t+1])*eps[t+1]
    return x

theta=np.array([0.2,0.5,0.3])
x=sim_garch(theta,np.ones(10000))

theta=np.random.random(3)
res=opt.minimize(garch_llh,theta,x)
np.exp(res.x)

from arch import arch_model
am = arch_model(y)
res=am.fit(update_freq=5)
print(res.summary())

import statsmodels.tsa.api as smt
import scipy.stats as scs

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return

_= tsplot(y, lags=30)

sig2=gen_garch_sig(np.log(np.array(res.params[1:])),y)
eps=y[1:]/np.sqrt(sig2[1:])
sm.qqplot(eps,line='45')
plt.show()