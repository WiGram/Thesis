import numpy as np
import pandas as pd
import genData as gd
from scipy import optimize as opt
import llhTestOnReturnSeriesMult as llhmult


class portfolio:
    def __init__(self,data=None):
        data = gd.genData()
        self.prices=data[0]
        self.monthlyRets=data[1]
        self.excessMRets=data[2]*12
        self.colNames=data[3]
        self.assets=data[4]
        self.monthlyVol=data[5]*np.sqrt(12)
        self.retCov=data[6]*12
        self.rf=data[7]
        self.pDates=data[8]
        self.rDates=data[9]
        
        # Historical moments and Sharpe Ratio
        self.mean_excess=self.excessMRets.mean()
        self.std_excess=self.excessMRets.std()
        self.sr_excess=self.pf_sharpe_fct(self.mean_excess,self.std_excess)
        
        # Table over historical moments and Sharpe Ratio
        description = pd.concat(
            [self.mean_excess,
             self.std_excess,
             self.sr_excess],
            axis=1,
            join_axes=[self.mean_excess.index]).transpose()
        description.index = ['Mean','Std','SR']
        self.description_uni=description
        
        # Historical Sharpe Ratio optimisation
        #  Shorting allowed
        results=self.market_pf()
        self.market_pf_SR=-results.fun
        self.market_pf_weights=results.x
        
        #  No shorting allowed
        results=self.constrained_market_pf()
        self.constrained_market_pf_SR=-results.fun
        self.constrained_market_pf_weights=results.x
        
        # Include exogenous regressor
        x  = pd.read_excel('/home/william/Dropbox/KU/K4/Python/DivYield.xlsx', 'Monthly')
        self.exogenous  = np.array(x.iloc[:,1])
        
        # Likelihood values
        self.mult_N_results = self.llhMult()
    
    def pf_return_fct(self,weights,returns):
        if len(weights) != returns.shape[0]:
            raise TypeError('Weights should have dimension (1x{}'.format(returns.shape[0]))
        return weights.dot(returns)
    
    def pf_std_fct(self,weights,cov):
        if len(weights) != cov.shape[1]:
            raise TypeError('Weights should have dimension ({}x1)'.format(cov.shape[1]))
        return np.sqrt(weights.dot(cov.dot(weights)))
    
    def pf_sharpe_fct(self,returns,std):
        return returns / std
    
    def portfolio_return(self,weights):
        pf_rets = self.pf_return_fct(weights,self.mean_excess)
        pf_standard_dev = self.pf_std_fct(weights,self.retCov)
        pf_sharpe_ratio = self.pf_sharpe_fct(pf_rets,pf_standard_dev)
        pf_table=pd.DataFrame([pf_rets,pf_standard_dev,pf_sharpe_ratio],
                               columns = ['Portfolio'],
                               index = self.description_uni.index)
        return pf_table
    
    def market_pf(self):
        weights=self.mk_weights()
        f = lambda w,x,y: - self.pf_return_fct(w,x) / self.pf_std_fct(w,y)
        args = self.mean_excess,self.retCov
        results = opt.minimize(f,weights,args)
        return results
    
    def check_sum(self,weights):
        '''
        Produces:
        -------------------------
        Returns 0 if individual weights sum to 1.0
        
        Motivation:
        -------------------------
        Applied as a constraint for opt.minimize.
        '''
        return np.sum(weights) - 1.0
    
    def mk_weights(self,a=None):
        if np.any(a==None):
            a=self.assets
            weights=np.random.random(a)
            weights/=np.sum(weights)
            return weights
        else:
            weights=np.random.random(a)
            weights/=np.sum(weights)
            return weights
    
    def constrained_market_pf(self,ret=None,cov=None):
        w=self.mk_weights()
        g=self.check_sum
        cons=({'type':'eq','fun': g})
        f=lambda w,x,y: - self.pf_return_fct(w,x) / self.pf_std_fct(w,y)
        if np.any(ret==None):
            assets=self.assets
            bnds=tuple(zip(np.zeros(assets),np.ones(assets)))
            args=self.mean_excess,self.retCov
            res=opt.minimize(f,w,args=args,bounds=bnds,constraints=cons)
            return res
        else:
            assets=cov.shape[0]
            bnds=tuple(zip(np.zeros(assets),np.ones(assets)))
            args=ret, cov
            res=opt.minimize(f,w,args=args,bounds=bnds,constraints=cons)
            return res
    
    def llhMult(self):
        # Renaming for brevity
        y=np.array(self.excessMRets.T)
        x=self.exogenous
        a=self.assets
        
        # Moments
        mu=self.mean_excess
        chol=np.linalg.cholesky(self.retCov)  # Cholesky decomp
        for i in range(self.assets):
            chol[i,i]=np.log(chol[i,i])  # Algorithm will take exp to diagon
        chol_idx=np.tril_indices(self.assets)  # Lower triangle only
        chol_pars=chol[chol_idx]
        
        # Optimisation method
        m = 'L-BFGS-B'
        
        # ===== Normal multivariate likelihood ====== #
        pars=np.concatenate((mu,chol_pars))
        
        llhmult.llhFct(pars,y)
        
        # Optimisation
        resN=opt.minimize(llhmult.llhFct,pars,y,method=m)
        
        # Information Criteria
        aicN,bicN,hqicN=llhmult.InfoCrit(pars,y,resN.fun)
        
        # ===== AR multilvariate likelihood ========= #
        ar=np.random.uniform(low=0.1, high=0.3, size=a)
        pars=np.concatenate((mu,ar,chol_pars))
        
        llhmult.llhFctAR(pars,y)  # empty calc to initialise numba
        
        # Optimisation
        resAR=opt.minimize(llhmult.llhFctAR, pars, y, method=m)
        
        # Information Criteria
        aicAR,bicAR,hqicAR=llhmult.InfoCrit(pars,y,resAR.fun)
        
        # ===== Exogenous multivariate likelihood === #
        ex=np.random.uniform(low=0.1, high=0.3, size=a)
        pars=np.concatenate((mu, ex, chol_pars))
        args=y,x
        
        llhmult.llhFctX(pars, *args)
        
        # Optimisation
        resX=opt.minimize(llhmult.llhFctX, pars, args=args, method=m)
        
        # Information Criteria
        aicX,bicX,hqicX=llhmult.InfoCrit(pars,y,resX.fun)
        
        # ===== Exogenous AR multivariate likelihood = #
        pars=np.concatenate((mu, ar, ex, chol_pars))
        
        llhmult.llhFctXAR(pars, *args)
        
        # Optimisation
        resXAR=opt.minimize(llhmult.llhFctXAR, pars, args=args, method=m)
        
        aicXAR,bicXAR,hqicXAR=llhmult.InfoCrit(pars,y,resXAR.fun)
        
        
        # ===== Summary statistics ================== #
        dic={'Normal':  [resN.fun,aicN,bicN,hqicN], 
             'AR(1)' :  [resAR.fun,aicAR,bicAR,hqicAR],
             'Exog.' :  [resX.fun,aicX,bicX,hqicX],
             'AR(1), Exog.' :   [resXAR.fun,aicXAR,bicXAR,hqicXAR]}
        idx=['Likelihood Value','AIC','BIC','HQIC']
        
        return pd.DataFrame(data=dic,index=idx)

opt.minimize(llhFct,pars,y,method=m)
pf = portfolio()
pf.description_uni
pf.market_pf_SR
pf.constrained_market_pf_SR
pf.market_pf_weights
pf.constrained_market_pf_weights

pf.constrained_market_pf(pf.mean_excess,pf.retCov)


"""
def __optimal_pf_return(self):
    weight_one = np.arange(1.01,step=0.01)
    weight_two = 1.0 - weight_one
    w_zip = list(zip(weight_one,weight_two))
    pf_rets = np.array([self.pf_return_fct(self.excess_return, np.array((x,y))) for x,y in w_zip])
    pf_std_dev = np.array([self.pf_std_fct(self.covariance, np.array((x,y))) for x,y in w_zip])
    pf_sr = self.pf_sharpe_fct(pf_rets,pf_std_dev)
    pf_cols = ["{}: {},{}: {}".format(self.tickers[0],x,self.tickers[1],y) for x,y in w_zip]
    pf_weights_table = pd.DataFrame([pf_rets,pf_std_dev,pf_sr],
                                            columns = pf_cols,
                                            index = self.table.index)
    return pf_weights_table
"""


"""
tickers = ['AAPL','GOOG']
np.random.seed(23456)
returns = np.random.normal(size = (len(tickers),100))
risk_free = 0.01
weights = np.array([0.5,0.5])

pf = portfolio(tickers,returns,risk_free)
pf.__dict__
pf.weights_table.idxmax(axis=1)

pf.weights_table[pf.weights_table.idxmax(axis=1)['Sharpe Ratio']]

weights = np.array


per=pf.pf_return_fct(pf.excess_return,weights)
psd=pf.pf_std_fct(pf.covariance,weights)
pf.pf_sharpe_fct(per,psd)

pf.table

pf_performance = pf.portfolio_return(weights=weights)
pf_performance

test = pf.optimal_pf_return()

test.idxmax(axis=1)

pf.pf_weights_table
"""

