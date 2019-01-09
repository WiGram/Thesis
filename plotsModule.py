from matplotlib import pyplot as plt   # Default plotting
import numpy as np
import matplotlib.mlab as mlab

plt.style.use('ggplot')                # Use grid from likes of R

def plotUno(x, y, yLab = 'Return process', xLab = 'Time', title = '', loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y, label = yLab)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotDuo(x, y1, y2, yLab1, yLab2, xLab, yLab, title = "", loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotTri(x, y1, y2, y3, xLab, yLab1, yLab2, yLab3, yLab, title = "", loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2)
    ax.plot(x, y3, label = yLab3)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotQuad(x, y1, y2, y3, y4, xLab, yLab1, yLab2, yLab3, yLab4, yLab, title = "", loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2)
    ax.plot(x, y3, label = yLab3)
    ax.plot(x, y4, label = yLab4)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def plotSVModel(x, y1, y2, yLab1, yLab2, yLab, xLab, title = '', loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y1, label = yLab1)
    ax.plot(x, y2, label = yLab2, marker = 'o', alpha = 0.5, markerfacecolor="None")
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def scatterUno(x, y, yLab = 'Return process', xLab = 'Lagged return process', title = '', loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(x, y, label = yLab, facecolors = 'none', edgecolors = 'r', alpha = 0.5)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def scatterDuo(x1, x2, y1, y2, yLab1, yLab2, yLab = 'Variance', xLab = 'Lagged returns', title = "", loc = 'lower right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.scatter(x1, y1, s=10, c='b', marker="s", label=yLab1)
    ax.scatter(x2, y2, s=10, c='r', marker="o", label=yLab2)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    ax.set_ylabel(yLab)
    ax.set_xlabel(xLab)
    fig.tight_layout()
    return plt.show()

def hist(x, bins = 50, density = True, title = '', loc = 'upper right'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.hist(x, bins = bins, density = density, facecolor = 'b', alpha = 0.75, rwidth=0.85)
    ax.set_title(title)
    ax.legend(loc = loc, shadow = False)
    fig.tight_layout()
    return plt.show()

def qqPlot(z, yLab = 'Empirical Quantiles', xLab = 'Theoretical Quantiles', title = 'Normal QQ-Plot'):
    x = np.sort(np.random.normal(size = len(z)))
    z = np.sort(np.squeeze(z))
    
    return scatterUno(x, z, yLab = yLab, xLab = xLab, title = title)