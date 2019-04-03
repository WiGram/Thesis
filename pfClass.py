import numpy as np
import pandas as pd
import genData as gd
from scipy import optimize as opt
import llhTestOnReturnSeriesMult as llhmult
import simulateSimsReturns as ssr
import matlab_results as mr
import constrainedOptimiser as copt
import expectedUtility947 as eu
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True) # scientific non-pandas
pd.options.display.float_format = '{:.4f}'.format # scientific pandas


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
        self.market_pf_results=self.market_pf()
        
        risk_free_split_market_pf = self.market_pf_split_uncons()
        
        un_mu=risk_free_split_market_pf[1]
        un_sd=risk_free_split_market_pf[2]
        un_sr=un_mu/un_sd
        
        un_weights=risk_free_split_market_pf[0]
        
        rf_weight=1-un_weights
        mp_weight=un_weights*self.market_pf_results.x
        
        un_weights=np.hstack((mp_weight,rf_weight))
        
        #  No shorting allowed
        self.constrained_market_pf_results=self.constrained_market_pf()
        
        risk_free_split_market_pf=self.market_pf_split_cons()
        
        con_mu=risk_free_split_market_pf[1]
        con_sd=risk_free_split_market_pf[2]
        con_sr=con_mu/con_sd
        
        con_weights=risk_free_split_market_pf[0]
        
        rf_weight=1-con_weights
        mp_weight=con_weights*self.constrained_market_pf_results.x
        
        con_weights=np.hstack((mp_weight,rf_weight))
        
        # Combining unconstrained and constrained (rows)
        d={'Unconstrained':(un_mu,un_sd,un_sr,*un_weights),
            'Constrained':(con_mu,con_sd,con_sr,*con_weights)}
        idx=('Return','Standard Deviation','Sharpe Ratio',
        'HY','IG','Comm','R2000','R1000','RF')
        
        df = pd.DataFrame(d,index=idx)
        self.optimal_sr_allocation=df
        
        # Utility optimisation; Non-market pf
        g=5.
        
        result_qdr=self.optUtility(gamma=g,f=self.quadraticUtility)
        self.opt_uncons_quad=result_qdr
        
        result_qdr=self.constrainedOptUtility(gamma=g,f=self.quadraticUtility)
        self.opt_cons_quad=result_qdr
        
        result_hpb=self.optUtility(gamma=g,f=self.hyperbolicUtility)
        self.opt_uncons_hyperbolic=result_hpb
        
        result_hpb=self.constrainedOptUtility(gamma=g,f=self.hyperbolicUtility)
        self.opt_cons_hyperbolic=result_hpb
        
        # Include exogenous regressor
        path='/home/william/Dropbox/KU/K4/Python/DivYield.xlsx'
        x  = pd.read_excel(path, 'Monthly')
        self.exogenous  = np.array(x.iloc[:,1])
        
        # Likelihood values
        self.mult_N_results = self.llhMult()
    
    def pf_return_fct(self,weights,returns):
        if len(weights) != returns.shape[0]:
            msg='Weights should have dimension (1x{}'.format(returns.shape[0])
            raise TypeError(msg)
        return weights.dot(returns)
    
    def pf_std_fct(self,weights,cov):
        if len(weights) != cov.shape[1]:
            msg='Weights should have dimension ({}x1)'.format(cov.shape[1])
            raise TypeError(msg)
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
    
    def market_pf_split_cons(self,c=0.5):
        weights=self.constrained_market_pf_results.x
        returns=self.pf_return_fct(weights,self.mean_excess)
        std=self.pf_std_fct(weights,self.retCov)
        sr=self.pf_sharpe_fct(returns,std)
        
        u=sr**2/(2*c)
        sd=sr/c
        mu=sr*sd
        
        opt_weight=mu/returns
        
        return opt_weight,mu,sd
        
    def market_pf_split_uncons(self,c=0.5):
        weights=self.market_pf_results.x
        returns=self.pf_return_fct(weights,self.mean_excess)
        std=self.pf_std_fct(weights,self.retCov)
        sr=self.pf_sharpe_fct(returns,std)
        
        u=sr**2/(2*c)
        sd=sr/c
        mu=sr*sd
        
        opt_weight=mu/returns
        
        return opt_weight,mu,sd
    
    def llhMult(self):
        # Renaming for brevity
        y=np.array(self.excessMRets.T/12)
        x=self.exogenous
        a=self.assets
        
        # Moments
        mu=self.mean_excess
        chol=np.linalg.cholesky(self.retCov/12)  # Cholesky decomp
        for i in range(self.assets):
            chol[i,i]=np.log(chol[i,i])  # Algorithm will take exp to diag
        chol_idx=np.tril_indices(self.assets)  # Lower triangle only
        chol_pars=chol[chol_idx]
        
        # Optimisation method
        m = 'L-BFGS-B'
        
        # ===== Normal multivariate likelihood ====== #
        pars=np.concatenate((mu,chol_pars))
        
        llhmult.llhFct(pars,y)
        
        # Optimisation
        resN=opt.minimize(llhmult.llhFct,pars,args=y,method=m)
        
        # Information Criteria
        aicN,bicN,hqicN=llhmult.InfoCrit(pars,y,resN.fun)
        
        # ===== AR multilvariate likelihood ========= #
        ar=np.random.uniform(low=0.1, high=0.3, size=a)
        pars=np.concatenate((mu,ar,chol_pars))
        
        llhmult.llhFctAR(pars,y)  # empty calc to initialise numba
        
        # Optimisation
        resAR=opt.minimize(llhmult.llhFctAR,pars,y,method=m)
        
        # Information Criteria
        aicAR,bicAR,hqicAR=llhmult.InfoCrit(pars,y,resAR.fun)
        
        # ===== Exogenous multivariate likelihood === #
        ex=np.random.uniform(low=0.1,high=0.3,size=a)
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
        resXAR=opt.minimize(llhmult.llhFctXAR,pars,args=args,method=m)
        
        aicXAR,bicXAR,hqicXAR=llhmult.InfoCrit(pars,y,resXAR.fun)
        
        # ===== Summary statistics ================== #
        dic={'Normal':  [resN.fun,aicN,bicN,hqicN], 
             'AR(1)' :  [resAR.fun,aicAR,bicAR,hqicAR],
             'Exog.' :  [resX.fun,aicX,bicX,hqicX],
             'AR(1), Exog.' :   [resXAR.fun,aicXAR,bicXAR,hqicXAR]}
        idx=['Likelihood Value','AIC','BIC','HQIC']
        
        return pd.DataFrame(data=dic,index=idx)
    
    def ultimateWealth(self,weight):
        # Cannot be 100 pct. in all assets
        # weight = self.mk_weights()
        w=weight
        
        # Compounded risk free rate; decimal representation
        c_rf = w[len(w)-1] * np.exp(np.sum(self.rf/100))
        
        # Compounded risky rate (non-excess); decimal representation
        c_risky = w[:len(w)-1] * np.exp(np.sum(self.monthlyRets/100))
        
        # Add the two compounded returns to final wealth
        wealth = c_rf + c_risky.sum()
        return wealth
    
    def hyperbolicUtility(self,weight,gamma):
        wealth = self.ultimateWealth(weight)
        utility = wealth**(1-gamma)/(1-gamma)
        return -utility
    
    def quadraticUtility(self,weight,gamma):
        wealth=self.ultimateWealth(weight)
        utility=wealth-gamma/2*wealth**2
        return -utility
    
    def optUtility(self,gamma,f):
        w = self.mk_weights(a=self.assets+1) # incl. risk free
        # e.g. f = hyperbolicUtility(weight,gamma)
        result=opt.minimize(f,w,args=gamma)
        return result
    
    def constrainedOptUtility(self,gamma,f):
        a = self.assets + 1
        w = self.mk_weights(a=a) # incl. risk free
        g=self.check_sum
        cons=({'type':'eq','fun': g})
        bnds=tuple(zip(np.zeros(a),np.ones(a)))
        args=gamma
        res=opt.minimize(f,w,args=args,bounds=bnds,constraints=cons)
        return res
    
    def simulate_model(
        self,states=2,sims=50000,mat=360,start=1,
        mu=mr.mu,cov=mr.cov,probs=mr.probs
    ):
        u = np.random.uniform(size=(sims,mat))
        self.sim_returns,self.sim_states=ssr.returnSim(
            states,sims,1,self.assets,start,mu,cov,probs,mat,u
        )
    
    def simulate_optimal_weights(self,rf=0.3,g=5.):
        a=self.assets+1
        w=np.random.random(a)
        w/=np.sum(w)
        #gamma=np.array([3,5,7,9,12])
        self.maturities=np.array([
            1,2,3,6,9,12,15,18,21,24,30,36,42,48,54,60,72,84,96,108,120,180,240,300,360
        ])
        self.opt_sim_weights=np.squeeze(list(zip(
            [np.repeat(w[i],len(self.maturities)) for i in range(len(w))]
        ))).T
        
        R = [self.sim_returns[:,:,:mat] for mat in self.maturities]
        
        #for g in gamma:
        for i,mat in enumerate(self.maturities):
            args=R[i],rf,g,self.assets,mat
            results=copt.constrainedOptimiser(
                eu.expectedUtilityMult,w,args,self.assets+1
            )
            self.opt_sim_weights[i]=results.x
        self.plot_simulate_optimal_weights()
    
    def plot_simulate_optimal_weights(self):
        labels=np.array(['hy','ig','cm','r2','r1','rf'])
        for i,lbl in enumerate(labels):
            plt.plot(
                self.maturities,self.opt_sim_weights[:,i],label=lbl
            )
        plt.ylim(top=1.0,bottom=0.0)
        plt.grid(b=True)
        plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc=0,
           ncol=6, mode="expand", borderaxespad=0.,
           fontsize=10)
        plt.show()


pf = portfolio()
pf.simulate_model()
pf.simulate_optimal_weights()


pf.market_pf_split_cons()
pf.optimal_sr_allocation
pf.description_uni
pf.market_pf_results
pf.constrained_market_pf_results

pf.constrained_market_pf(pf.mean_excess,pf.retCov)


"""
def __optimal_pf_return(self):
    weight_one = np.arange(1.01,step=0.01)
    weight_two = 1.0 - weight_one
    w_zip = list(zip(weight_one,weight_two))
    pf_rets=np.array([self.pf_return_fct(self.excess_return,np.array((x,y))) for x,y in w_zip])
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

