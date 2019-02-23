import plotsModule as pltm
from matplotlib import pyplot as plt
import numpy as np

# sharex = True: the x-axis will be the same for all.. sharey = True is also possible

# emPlots(sims, states, assets, rDates, colNames, llh, ps, vs, ms, pStar)

def emPlots(sims, states, assets, rDates, colNames, 
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

    # Volatility plots
    if regPlot == True:
        fig, axes = plt.subplots(nrows = 1,ncols = states,sharex = True,figsize = (15,6))
        test = np.array([pStar[j,:] for j in range(states)])
        for ax, title, y in zip(axes.flat, stateTitle, test):
            ax.plot(rDates, y)
            ax.set_title(title)
            ax.grid(False)
        plt.show()
    
    # Transition probabilities
    if probPlot == True:
        for i in range(states):
            plt.plot(ps[:,i,i], label = "p{}{}".format(i+1,i+1))
        plt.legend()
        plt.ylabel('Probabilitiy')
        plt.xlabel('Trials')
        plt.show()

    # Return volatility plot
    if volPlot == True:
        for j, txt in zip(range(states), volTitle):
            fig, axes = plt.subplots(nrows = 3, 
                                    ncols = 2, 
                                    sharex = True, 
                                    figsize = (15,15))
            fig.suptitle(txt, fontsize=16)

            test = np.array([vs[:,j,i,i] for i in range(assets)])
            for ax, title, y in zip(axes.flat, colNames, test):
                ax.plot(range(sims), y)
                ax.set_title(title)
                ax.grid(False)
            plt.show()
    
    # Mean returns plot
    if retPlot == True:
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



