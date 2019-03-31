import numpy as np
import pandas as pd

class portfolio:
    def __init__(self,tickers,returns,risk_free):
        self.tickers=tickers
        self.returns=returns
        self.risk_free=risk_free
        self.excess_return=np.mean(self.returns-self.risk_free, axis = 1)
        self.standard_dev=np.std(self.returns,axis = 1)
        self.covariance=np.cov(self.returns)
        self.sharpe_ratio=self.excess_return/self.standard_dev
        self.table=pd.DataFrame([self.excess_return,
                                   self.standard_dev,
                                   self.sharpe_ratio],
                                  columns = tickers,
                                  index = ['Mean Excess Return',
                                           'Standard deviation',
                                           'Sharpe Ratio'])
        #self.weights_table = self.__optimal_pf_return()
    
    def pf_return_fct(self,returns,weights):
        return weights.dot(returns)
    
    def pf_std_fct(self,cov,weights):
        return np.sqrt(weights.dot(cov.dot(weights)))
    
    def pf_sharpe_fct(self,returns,std):
        return returns / std
    
    def portfolio_return(self,weights):
        pf_rets = self.pf_return_fct(self.excess_return,weights)
        pf_standard_dev = self.pf_std_fct(self.covariance,weights)
        pf_sharpe_ratio = self.pf_sharpe_fct(pf_rets,pf_standard_dev)
        pf_table=pd.DataFrame([pf_rets,pf_standard_dev,pf_sharpe_ratio],
                               columns = ['{}: {}, {}: {}'.format(self.tickers[0],weights[0],
                                                                  self.tickers[1],weights[1])],
                               index = self.table.index)
        return pf_table
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

2*3