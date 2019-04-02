
"""
mu1 = np.array([0.24])
mu2 = np.array([0.12])
mu  = np.concatenate((mu1,mu2))

cov = np.array([[0.14,-0.21],
                [-0.21,0.07]])

returns = np.random.multivariate_normal(mu,cov,size = 60)

w = np.array([0.3,0.4])
g = 5
args = returns, g

ApB = 3
gamma = 5

args = returns
bnds=tuple(zip(np.zeros(ApB),np.ones(ApB)))
cons=({'type':'eq','fun': check_sum})


opt.minimize(expUtil, w, args = args,bounds=bnds,constraints=cons)

w  = np.random.random(size = (2,200))
ww = np.sum(w, axis = 0)
w  = w / ww
w  = w.T

utils = np.zeros(200)
for i in range(len(utils)):
    utils[i] = expUtil(w[i], returns, gamma)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(w[:,0],w[:,1],utils,c='r',marker='o')
ax.set_xlabel('High Yield allocation')
ax.set_ylabel('Russell 1000 allocation')
ax.set_zlabel('Expected Utility')
plt.show()

plt.scatter(w[:,1],utils)
plt.show()
"""
