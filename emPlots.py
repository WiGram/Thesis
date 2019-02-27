import plotsModule as pltm
from matplotlib import pyplot as plt
import numpy as np

# sharex = True: the x-axis will be the same for all.. sharey = True is also possible

# emPlots(sims, states, assets, rDates, colNames, llh, ps, vs, ms, pStar)

def emPlots(sims, states, assets, rDates, colNames, 
            llh, ps, vs, ms, pStar, 
            plots = 'all'):
    stateTitle = ['State '+i for i in map(str,range(1, states + 1))]
    volTitle = ['Volatility, state ' + i for i in map(str, range(1, states + 1))]
    retTitle = ['Return, state ' + i for i in map(str, range(1, states + 1))]

    # Log-likelihood convergence plot
    if plots == 'all' or plots == 'likelihood':
        pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

    # Smoothed probabilities
    if plots == 'all' or plots == 'smoothed':
        fig, axes = plt.subplots(nrows = 1,ncols = states,sharex = True,figsize = (15,6))
        test = np.array([pStar[j,:] for j in range(states)])
        for ax, title, y in zip(axes.flat, stateTitle, test):
            ax.plot(rDates, y)
            ax.set_title(title)
            ax.grid(False)
        plt.show()
    
    # Transition probabilities
    if plots == 'all' or plots == 'transition':
        for i in range(states):
            plt.plot(ps[:,i,i], label = "p{}{}".format(i+1,i+1))
        plt.legend()
        plt.ylabel('Probability')
        plt.xlabel('Trials')
        plt.show()

    # Return volatility plot
    if plots == 'all' or plots == 'vol':
        for j, txt in zip(range(states), volTitle):
            fig, axes = plt.subplots(nrows = 3, 
                                    ncols = 2, 
                                    sharex = True, 
                                    figsize = (15,15))
            fig.suptitle(txt, fontsize=16)

            test = np.array([np.sqrt(vs[:,j,i,i]) for i in range(assets)])
            for ax, title, y in zip(axes.flat, colNames, test):
                ax.plot(range(sims), y)
                ax.set_title(title)
                ax.grid(False)
            plt.show()
    
    # Mean returns plot
    if plots == 'all' or plots == 'mu':
        for j, txt in zip(range(states), retTitle):
            fig, axes = plt.subplots(nrows = 3, 
                                    ncols = 2, 
                                    sharex = True, 
                                    figsize = (15,15))
            fig.suptitle(txt, fontsize=16)

            test = np.array([ms[:,i,j] for i in range(assets)])
            for ax, title, y in zip(axes.flat, colNames, test):
                ax.plot(range(sims), y)
                ax.set_title(title)
                ax.grid(False)
            plt.show()

    if plots not in ('all','likelihood','smoothed','transition','vol','mu'):
        raise ValueError('plots must be "all", "likelihood", "smoothed", "transition", "vol" or "mu"')


def emUniPlots(sims, states, rDates, colNames, 
            llh, ps, vs, ms, pStar, 
            llhPlot = True, regPlot = True, 
            probPlot = True, volPlot = True, 
            retPlot = True):
    stateTitle = ['State '+i for i in map(str,range(1, states + 1))]
    volTitle = ['Volatility, state ' + i for i in map(str, range(1, states + 1))]
    retTitle = ['Return, state ' + i for i in map(str, range(1, states + 1))]

    # Log-likelihood convergence plot
    if llhPlot == True:
        pltm.plotUno(range(sims), llh, yLab = 'log-likelihood value')

    # Smoothed probabilities
    if regPlot == True:
        fig, axes = plt.subplots(nrows = 1,ncols = states,sharex = True,figsize = (15,6))
        for ax, title, y in zip(axes.flat, stateTitle, pStar):
            ax.plot(rDates, y)
            ax.set_title(title)
            ax.grid(False)
        plt.show()
    
    # Transition probabilities
    if probPlot == True:
        for i in range(states):
            plt.plot(ps[:,i,i], label = "p{}{}".format(i+1,i+1))
        plt.legend()
        plt.ylabel('Probability')
        plt.xlabel('Trials')
        plt.show()

    # Return volatility plot
    if volPlot == True:
        for i in range(states):
            plt.plot(np.sqrt(vs[:,i]), label = "Volatility (s = {})".format(i+1))
        plt.legend()
        plt.ylabel('Volatility')
        plt.xlabel('Trials')
        plt.show()
    
    # Mean returns plot
    if retPlot == True:
        for i in range(states):
            plt.plot(ms[:,i], label = "Return (s = {})".format(i+1))
        plt.legend()
        plt.ylabel('Rate of return')
        plt.xlabel('Trials')
        plt.show()


