# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Inez Wijnands s4149696
# SML ASS 2 exercise1.py

from __future__ import division
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal

if __name__ == "__main__":
    np.random.seed(0)
    sigma = np.matrix('0.14 -0.3 0.0 0.2; -0.3 1.16 0.2 -0.8; 0.0 0.2 1.0 1.0; 0.2 -0.8 1.0 2.0')
    labda = np.linalg.inv(sigma)
    mu = np.matrix('1; 0; 1; 2')
    sigma_p = np.linalg.inv(labda[0:2,0:2])
    mu_p = mu[0:2] - sigma_p * labda[0:2,2:4] * (np.matrix('0;0') - mu[2:4])
    
    mu_t = np.random.multivariate_normal(np.array(mu_p).flatten(), sigma_p)
    print mu_t
    
    x,y = np.meshgrid(np.linspace(-0.5,2,250), np.linspace(-0.5,2,250))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    var = multivariate_normal(mean = np.array(mu_p).flatten(), cov = sigma_p)
    #zs = np.array([(xs**2 + ys) for xs,ys in zip(np.ravel(x), np.ravel(y))])
    #z = zs.reshape(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y, var.pdf(pos),cmap=cm.rainbow)
    ax.view_init(elev=35,azim=60)
    
    