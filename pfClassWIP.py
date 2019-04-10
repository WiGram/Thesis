import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import genData as gd
from scipy import optimize as opt
from llhTestOnReturnSeriesMult import llhFct,llhFctAR,llhFctX,llhFctXAR,InfoCrit
import simulateSimsReturns as ssr
import matlab_results as mr
import constrainedOptimiser as copt
import expectedUtility947 as eu
import matplotlib.pyplot as plt
np.set_printoptions(suppress = True) # scientific non-pandas
pd.options.display.float_format = '{:.4f}'.format # scientific pandas


class portfolio:
    def __init__(self,data=None):
        
        # Initial data generation
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
        # End of initial data generation
        
        # Include exogenous regressor
        # path='/home/william/Dropbox/KU/K4/Python/DivYield.xlsx'
        path='C:/Users/willi/Dropbox/KU/K4/Python/DivYield.xlsx'
        x  = pd.read_excel(path, 'Monthly')
        self.exogenous  = np.array(x.iloc[:,1])
        # End of inclusion of exogenous
        
        # Historical moments and Sharpe Ratio
        self.mean_excess=self.excessMRets.mean()
        self.std_excess=self.excessMRets.std()
        self.sr_excess=self.__pf_sharpe_fct(self.mean_excess,self.std_excess)
        # End of moments computation
        
        # Likelihood values
        self.mult_N_results = self.__llhMult()
        # End of likelihood values
        
        # Table over historical moments and Sharpe Ratio
        description = pd.concat(
            [self.mean_excess,
             self.std_excess,
             self.sr_excess],
            axis=1,
            join_axes=[self.mean_excess.index]).transpose()
        description.index = ['Mean','Std','SR']
        self.description_univariates=description
        # End of moments table
        
        # Optimal weights in market portfolio based on Sharpe Ratio optim.
        self.opt_market_pf=self.__market_pf(cons=False)
        self.opt_cons_market_pf=self.__market_pf(cons=True)
        # End of optimal market PF
        
        self.optimal_sr_allocation=self.__optimal_sr_allocation()
        
        self.utility_table=self.__utility_fcts()
        
        self.sim_returns={'States=2':{},'States=3':{}}
        self.sim_states={'States=2':{},'States=3':{}}
        
        self.opt_weights_unbounded={
            'States=2':{
                'Start=1':{},
                'Start=2':{}
            },
            'States=3':{
                'Start=1':{},
                'Start=2':{},
                'Start=3':{}
            }
        }
        self.opt_weights_bounded={
            'States=2':{
                'Start=1':{},
                'Start=2':{}
            },
            'States=3':{
                'Start=1':{},
                'Start=2':{},
                'Start=3':{}
            }
        }
    
    def __optimal_sr_allocation(self):
        """
        Computes weight allocation between the market portfolio
        and the risk-free asset.
        
        Assumes quadratic utility function:
        u = mu - c/2 *sd^2
        
        Two computations are done:
        1. Shorting allowed, weights must sum to 1.0
        2. Shorting not allowed, weights must sum to 1.0
        """
        
        # --- The unconstrained allocation --- #
        un_weights,un_mu,un_sd,un_sr = self.__market_pf_split(cons=False)

        rf_weight=1-un_weights
        mp_weight=un_weights*self.opt_market_pf.x
        
        # Final allocation
        un_weights=np.hstack((mp_weight,rf_weight))
        
        # --- Constrained allocation --- #
        con_weights,con_mu,con_sd,con_sr=self.__market_pf_split(cons=True)
        
        rf_weight=1-con_weights
        mp_weight=con_weights*self.opt_cons_market_pf.x
        
        # Final allocation
        con_weights=np.hstack((mp_weight,rf_weight))
        # --- End of allocations, constrained and unconstrained --- #
        
        # Constraints are bounds on portfolio weights (rows)
        d={'Unbounded':(un_mu,un_sd,un_sr,*un_weights),
            'Bounded':(con_mu,con_sd,con_sr,*con_weights)}
        idx=('Return','Standard Deviation','Sharpe Ratio',
        'HY','IG','Comm','R2000','R1000','RF')
        
        df = pd.DataFrame(d,index=idx)
        return df
    
    def __utility_fcts(self,g=5.0):
        # Utility optimisation; Non-market pf
        uncons_qdr=self.__optUtility(gamma=g,f=self.__quadraticUtility)
        cons_qdr=self.__optUtility(gamma=g,f=self.__quadraticUtility,bnd=True)
        uncons_hpb=self.__optUtility(gamma=g,f=self.__hyperbolicUtility)
        cons_hpb=self.__optUtility(gamma=g,f=self.__hyperbolicUtility,bnd=True)
        
        data = np.array([
            np.array([uncons_qdr.fun,*uncons_qdr.x]),
            np.array([cons_qdr.fun,*cons_qdr.x]),
            np.array([uncons_hpb.fun,*uncons_hpb.x]),
            np.array([cons_hpb.fun,*cons_hpb.x])
        ])
        idx  = (
            'Unconstrained Quadratic',
            'Constrained Quadratic',
            'Unconstrained Hyperbolic',
            'Constrained Hyperbolic'
        )
        cols = ('Utility','HY','IG','Comm.','R2000','R1000','RF')
        df=DataFrame(data,index=idx,columns=cols)
        return df
    
    def __pf_return_fct(self,weights,returns):
        return weights.dot(returns)
    
    def __pf_std_fct(self,weights,cov):
        return np.sqrt(weights.dot(cov.dot(weights)))
    
    def __pf_sharpe_fct(self,returns,std):
        return returns / std
    
    def portfolio_return(self,weights):
        pf_rets = self.__pf_return_fct(weights,self.mean_excess)
        pf_standard_dev = self.__pf_std_fct(weights,self.retCov)
        pf_sharpe_ratio = self.__pf_sharpe_fct(pf_rets,pf_standard_dev)
        pf_table=pd.DataFrame([pf_rets,pf_standard_dev,pf_sharpe_ratio],
                               columns = ['Portfolio'],
                               index = self.description_univariates.index)
        return pf_table
    
    def __market_pf(self, cons=False):
        """
        Computes optimal allocation within the market portfolio,
        shorting is allowed.
        
        Called:
        ------------------
        In initialisation
        """
        w=self.__mk_weights()
        cons=({'type':'eq','fun': self.__check_sum})
        f = lambda w,x,y: - self.__pf_return_fct(w,x) / self.__pf_std_fct(w,y)
        args = self.mean_excess,self.retCov
        if cons == False:
            results = opt.minimize(f,w,args,constraints=cons)
            return results
        else:
            bnds=tuple(zip(np.zeros(self.assets),np.ones(self.assets)))
            results=opt.minimize(f,w,args=args,bounds=bnds,constraints=cons)
            return results
    
    def __check_sum(self,weights):
        '''
        Produces:
        -------------------------
        Returns 0 if individual weights sum to 1.0
        
        Motivation:
        -------------------------
        Applied as a constraint for opt.minimize - weights sum to 1.
        
        Used in the following functions:
        -------------------------
        __market_pf()
        __OptUtility()
        '''
        return np.sum(weights) - 1.0
    
    def __mk_weights(self,a=None):
        if np.any(a==None):
            a=self.assets
            weights=np.random.random(a)
            weights/=np.sum(weights)
            return weights
        else:
            weights=np.random.random(a)
            weights/=np.sum(weights)
            return weights
    
    def __market_pf_split(self,c=0.5,cons=False):
        if cons == False:
            weights=self.opt_market_pf.x
        else:
            weights=self.opt_cons_market_pf.x
        
        returns=self.__pf_return_fct(weights,self.mean_excess)
        std=self.__pf_std_fct(weights,self.retCov)
        sr=self.__pf_sharpe_fct(returns,std)
        
        #u=sr**2/(2*c)
        sd=sr/c
        mu=sr*sd
        
        opt_weight=mu/returns
        
        return opt_weight,mu,sd,sr
    
    def __llhMult(self):
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
        
        # Optimisation
        resN=opt.minimize(llhFct,pars,args=y,method=m)
        
        # Information Criteria
        aicN,bicN,hqicN=InfoCrit(pars,y,resN.fun)
        
        # ===== AR multilvariate likelihood ========= #
        ar=np.random.uniform(low=0.1, high=0.3, size=a)
        pars=np.concatenate((mu,ar,chol_pars))
        
        # Optimisation
        resAR=opt.minimize(llhFctAR,pars,y,method=m)
        
        # Information Criteria
        aicAR,bicAR,hqicAR=InfoCrit(pars,y,resAR.fun)
        
        # ===== Exogenous multivariate likelihood === #
        ex=np.random.uniform(low=0.1,high=0.3,size=a)
        pars=np.concatenate((mu, ex, chol_pars))
        args=y,x
        
        # Optimisation
        resX=opt.minimize(llhFctX,pars,args=args,method=m)
        
        # Information Criteria
        aicX,bicX,hqicX=InfoCrit(pars,y,resX.fun)
        
        # ===== Exogenous AR multivariate likelihood = #
        pars=np.concatenate((mu, ar, ex, chol_pars))
        
        # Optimisation
        resXAR=opt.minimize(llhFctXAR,pars,args=args,method=m)
        
        aicXAR,bicXAR,hqicXAR=InfoCrit(pars,y,resXAR.fun)
        
        # ===== Summary statistics ================== #
        dic={'Normal':  [resN.fun,aicN,bicN,hqicN],
             'AR(1)' :  [resAR.fun,aicAR,bicAR,hqicAR],
             'Exog.' :  [resX.fun,aicX,bicX,hqicX],
             'AR(1), Exog.' :   [resXAR.fun,aicXAR,bicXAR,hqicXAR]}
        idx=['Likelihood Value','AIC','BIC','HQIC']
        
        return pd.DataFrame(data=dic,index=idx)
    
    def __ultimateWealth(self,weight):
        # Cannot be 100 pct. in all assets
        # weight = self.__mk_weights()
        w=weight
        
        # Compounded risk free rate; decimal representation
        c_rf = w[len(w)-1] * np.exp(np.sum(self.rf/100))
        
        # Compounded risky rate (non-excess); decimal representation
        c_risky = w[:len(w)-1] * np.exp(np.sum(self.monthlyRets/100))
        
        # Add the two compounded returns to final wealth
        wealth = c_rf + c_risky.sum()
        return wealth
    
    def __hyperbolicUtility(self,weight,gamma):
        wealth = self.__ultimateWealth(weight)
        utility = wealth**(1-gamma)/(1-gamma)
        return -utility
    
    def __quadraticUtility(self,weight,gamma):
        wealth=self.__ultimateWealth(weight)
        utility=wealth-gamma/2*wealth**2
        return -utility
    
    def __optUtility(self,gamma,f,bnd=False):
        w = self.__mk_weights(a=self.assets+1) # incl. risk free
        cons=({'type':'eq','fun': self.__check_sum})
        if bnd==False:
            # e.g. f = hyperbolicUtility(weight,gamma)
            result=opt.minimize(f,w,args=gamma,constraints=cons)
        else:
            bnds=tuple(zip(np.zeros(self.assets+1),np.ones(self.assets+1)))
            result=opt.minimize(f,w,args=gamma,bounds=bnds,constraints=cons)
        return result
    
    def simulate_model(
        self,states=2,sims=50000,mat=360,start=1,
        mu=mr.mu,cov=mr.cov,probs=mr.probs
    ):
        u = np.random.uniform(size=(sims,mat))
        sim_returns,sim_states=ssr.returnSim(
            states,sims,1,self.assets,start,mu,cov,probs,mat,u
        )
        
        self.sim_returns[
            'States={}'.format(states)
        ][
            'Start={}'.format(start)
        ]=sim_returns
        
        self.sim_states[
            'States={}'.format(states)
        ][
            'Start={}'.format(start)
        ]=sim_states
    
    def sim_opt_weights(
        self,rf=0.3,g=5.,bnd=True,
        states=2,sims=50000,mat=360,start=1,
        mu=mr.mu,cov=mr.cov,probs=mr.probs
    ):
        if len(self.sim_returns[
            'States={}'.format(states)][
                'Start={}'.format(start)
            ])==0:
                self.simulate_model(
                    states=states,sims=sims,mat=mat,start=start,
                    mu=mr.mu,cov=mr.cov,probs=mr.probs
                )
                rets=self.sim_returns[
                    'States={}'.format(states)
                ][
                    'Start={}'.format(start)
                ]
        else:
            rets=self.sim_returns[
                'States={}'.format(states)
            ][
                'Start={}'.format(start)
            ]
        
        a=self.assets+1
        w=np.random.random(a)
        w/=np.sum(w)
        #gamma=np.array([3,5,7,9,12])
        maturities=np.array([
            1,2,3,6,9,12,
            15,18,21,24,
            30,36,42,48,
            54,60,72,84,
            96,108,120
        ])
        # labels=np.array(['hy','ig','cm','r2','r1','rf'])
        labels=np.hstack((self.colNames,'Risk Free'))
        weights=np.squeeze(list(zip(
            [np.repeat(w[i],len(maturities)) for i in range(len(w))]
        ))).T
        
        R = [rets[:,:,:mat] for mat in maturities]
        
        if bnd:
            #for g in gamma:
            for i,mat in enumerate(maturities):
                args=R[i],rf,g,self.assets,mat
                results=copt.boundedOptimiser(
                    eu.expectedUtilityMult,w,args,a
                )
                weights[i]=results.x
            for i,asset in enumerate(labels):
                try:
                    self.opt_weights_bounded[
                        'States={}'.format(states)
                    ][
                        'Start={}'.format(start)
                    ][
                        asset
                    ][
                        'gamma={}'.format(g)
                    ] = weights[:,i]
                except:
                    self.opt_weights_bounded[
                        'States={}'.format(states)
                    ][
                        'Start={}'.format(start)
                    ][
                        asset
                    ]=pd.DataFrame(
                        weights[:,i],columns=['gamma={}'.format(g)],index=maturities
                    )
        else:
            for i,mat in enumerate(maturities):
                args=R[i],rf,g,self.assets,mat
                results=copt.unboundedOptimiser(
                    eu.expectedUtilityMult,w,args,a
                )
                weights[i]=results.x
            for i,asset in enumerate(labels):
                try:
                    self.opt_weights_unbounded[
                        'States={}'.format(states)
                    ][
                        'Start={}'.format(start)
                    ][
                        asset
                    ][
                        'gamma={}'.format(g)
                    ] = weights[:,i]
                except:
                    self.opt_weights_unbounded[
                        'States={}'.format(states)
                    ][
                        'Start={}'.format(start)
                    ][
                        asset
                    ]=pd.DataFrame(
                        weights[:,i],columns=['gamma={}'.format(g)],index=maturities
                    )


pf = portfolio()
pf.simulate_model()
for s in range(1,3):
    for g in np.array([3,5,7,9]):
        pf.sim_opt_weights(start=s,g=g)

"""
pf.simulate_model()
#pf.sim_opt_weights(bnd=True)

def plot_simulated_returns(self,returns):
    returns=pf.sim_returns
    labels=np.array(['hy','ig','cm','r2','r1'])
    titles=np.array([
        'High Yield','Investment Grade','Commodities',
        'Russell 2000','Russell 1000'
    ])
    dic={}
    for i,l in enumerate(labels):
        dic[l]=DataFrame(
            returns[:1000,i,:]*12
        )

    fig,axes=plt.subplots(
        nrows=3,ncols=2,sharex=True,figsize=(15,15)
    )
    for ax,l,t in zip(axes.flat,labels,titles):
        ax.set_title(t)
        ax.plot(
            dic[l].T,color='grey',alpha=0.3
        )
        ax.plot(
            dic[l].iloc[25,:],color='blue',alpha=.7
        )
        ax.plot(
            dic[l].quantile(.025),color='black',linestyle='dashed',alpha=.7
        )
        ax.plot(
            dic[l].quantile(.975),color='black',linestyle='dashed',alpha=.7
        )
        ax.plot(np.array(pf.excessMRets.iloc[:360,i]))
    plt.show()
"""
