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

# Estimate ARCH models
from arch import arch_model

# Draw TS plots 
# (https://github.com/Auquan/Tutorials/blob/master/Time%20Series%20Analysis%20-%204.ipynb)
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

from pfClassWIP import portfolio
pf=portfolio()
y=np.array(pf.exogenous)
dy=y[1:]-y[:len(y)-1]
_=tsplot(y,lags=30) #Heavy tails including possible volatility clustering -> arch/garch
_=tsplot(dy,lags=30)

""" Functions to produce sigma^2 and log likelihood values """
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

def sim_garch(theta,y):
    m=len(y)
    o,a,b=np.exp(theta)
    eps=np.random.normal(size=m)
    sig2=np.zeros(m)
    x=np.zeros(m)
    for t in range(m-1):
        sig2[t+1]=o+a*x[t]**2+b*sig2[t]
        x[t+1]=np.sqrt(sig2[t+1])*eps[t+1]
    return x

""" We first try on a manufactured process """
theta=np.log(np.array([0.2,0.5,0.3]))
x=sim_garch(theta,np.ones(10000))
_ = tsplot(x, lags=30) # no significant lags, heavy tails as expected

""" Estimating the model using the ARCH-package """
am = arch_model(x)
res=am.fit(update_freq=5)
print(res.summary())

theta=np.array(res.params[1:])
sig2=gen_garch_sig(np.log(theta),x)
eps=x[1:]/np.sqrt(sig2[1:])
sm.qqplot(eps,line='45')
plt.show()
""" So now we have confirmed the simulation and estimation works"""

""" repeat for actual data """
_ = tsplot(dy/np.std(dy),lags=30) # lags are larger here, than for x, other: the same
am=arch_model(dy)
res=am.fit(update_freq=5)
print(res.summary()) 

theta=np.array(res.params[1:])
sig2=gen_garch_sig(np.log(theta),dy)
eps=dy[1:]/np.sqrt(sig2[1:])
sm.qqplot(eps,line='45')
plt.show() # Doing better, but not completely there yet, with heavy tails.
""" Tails are too heavy, consider GARCH(2,2) """

""" GARCH(2,2): Functions to produce sigma^2 and log likelihood values """
def gen_garch2_sig(theta,y):
    o,a1,a2,b1,b2=np.exp(theta)
    sig2=np.zeros(len(y))
    for t in range(len(y)-2):
        sig2[t+2]=o+a1*y[t+1]**2+a2*y[t]**2+b1*sig2[t+1]+b2*sig2[t]
    return sig2

def garch2_llh(theta,y):
    sig2=gen_garch2_sig(theta,y)
    llh=-.5*(np.log(2*np.pi)+np.log(sig2[2:])+y[2:]**2/sig2[2:])
    return-np.sum(llh)

def sim_garch2(theta,y):
    m=len(y)
    o,a1,a2,b1,b2=np.exp(theta)
    eps=np.random.normal(size=m)
    sig2=np.zeros(m)
    x=np.zeros(m)
    for t in range(m-2):
        sig2[t+2]=o+a1*x[t+1]**2+a2*x[t]**2+b1*sig2[t+1]+b2*sig2[t]
        x[t+2]=np.sqrt(sig2[t+2])*eps[t+2]
    return x

theta=np.log(np.array([0.1,0.15,0.12,0.21,0.18]))
x=sim_garch2(theta,dy)
_=tsplot(x,lags=30) # looks like a mess.... probably won't work

am=arch_model(x,p=2,q=2)
res=am.fit(update_freq=5)
res.summary()
""" Consider only the second half of the sample """
am=arch_model(dy,p=2,q=2)
res=am.fit(update_freq=5)
theta=np.array(res.params[1:])
sig2=gen_garch2_sig(np.log(theta),dy)
eps=dy[2:]/np.sqrt(sig2[2:])
sm.qqplot(eps,line='45')
plt.show()
sm.qqplot(res.resid/np.std(dy),line='45')
plt.show() # better, but no cigar

res2=opt.minimize(garch2_llh,theta,dy,method='L-BFGS-B')
theta2=np.exp(res2.x)
sig2=gen_garch2_sig(np.log(theta2),dy)
eps=dy[2:]/np.sqrt(sig2[2:])
sm.qqplot(eps,line='45')
plt.show()

y.min()
y.argmin()
z=dy[y.argmin():]
am=arch_model(z)
res=am.fit(update_freq=5)
print(res.summary())
_ = tsplot(z/np.std(z), lags=30)

theta=np.array(res.params[1:])
sig2=gen_garch_sig(np.log(theta),z)
eps=z[1:]/np.sqrt(sig2[1:])
sm.qqplot(eps,line='45')
plt.show()

theta=np.random.random(5)
am=arch_model(z,p=2,q=2)
res=am.fit(update_freq=5)
res.summary()
theta=res.params[1:]
sig2=gen_garch2_sig(np.log(theta),z)
eps=z[2:]/np.sqrt(sig2[2:])
sm.qqplot(eps,line='45')
plt.show()

""" Estimate instead simple ARCH model """

theta=np.log(np.array([0.2,0.5]))
res=opt.minimize(arch_llh,theta,z)
theta=np.exp(res.x) # alpha=1.95 -> variance not defined, but still stationary
sig2=gen_arch_sig(np.log(theta),z)
eps=z[1:]/np.sqrt(sig2)
sm.qqplot(eps,line='45')
plt.show()
# Equivalent:
sm.qqplot(res.resid/np.std(z),line='45')
plt.show()

""" Since only the ARCH(1)-coefficient was significant: test AR(1)-ARCH(1)"""
def gen_arch_sig(theta,z):
    o,a=np.exp(theta)
    sig2=o + a*z[:len(z)-1]**2
    return sig2

def arch_llh(theta,z):
    sig2=gen_arch_sig(theta,z)
    llh=-.5*(np.log(2*np.pi)+np.log(sig2)+z[1:]**2/sig2)
    return - np.sum(llh)

def ar_arch_llh(theta,z):
    sig2=gen_arch_sig(theta[1:],z)
    mean=z[1:]-theta[0]*z[:len(z)-1]
    llh=-.5*(np.log(2*np.pi)+np.log(sig2)+mean**2/sig2)
    return -np.sum(llh)

theta=np.random.random(3)
ar_arch_llh(theta,z)
arch_llh(theta[1:],z)
res=opt.minimize(ar_arch_llh,theta,z,method='L-BFGS-B')
theta=res.x[0],np.exp(res.x[1:])
theta=np.hstack(theta)
theta

sig2=gen_arch_sig(theta[1:],z)
eps=(z[1:]-theta[0]*z[:len(z)-1])/np.sqrt(sig2)
sm.qqplot(eps,line='45')
plt.show()

def sim_ar_arch(theta,z):
    m=len(z)
    b,o,a=theta
    eps=np.random.normal(size=m)
    sig2=np.zeros(m)
    x=np.zeros(m)
    for t in range(m-1):
        sig2[t+1]=o+a*x[t]**2
        x[t+1]=b*x[t]+np.sqrt(sig2[t+1])*eps[t+1]
    return x

theta=np.random.random(3)
theta
x=sim_ar_arch(theta,z)
res=opt.minimize(ar_arch_llh,theta,x,method='L-BFGS-B')
res.x[0],np.exp(res.x[1:])

am=arch_model(x,p=5,q=0)
res=am.fit(update_freq=5)
res.plot()
sm.qqplot(res.resid/np.std(res.resid),line='45')
plt.show()
res=opt.minimize(arch_llh,theta[1:],x,method='L-BFGS-B')