# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 4 exercise2.py
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D



#Gaussian
def ex1_gaussian(x):
    sigma = 0.4 * np.identity(2)

    return 3 * multivariate_normal.pdf(x,[0,0],sigma)

def plot_gaussian(data_function, a_space, b_space, name):
    X, Y = np.meshgrid(a_space, b_space)
        
    z = np.array( [data_function([x,y]) for x,y in zip(np.ravel(X),np.ravel(Y))])
    Z = z.reshape(X.shape)
    
    #Plot
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    #plt.title("Isotropic 2D Gaussian")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    surf = ax.plot_surface(X,Y, Z, cmap=cm.winter, linewidth=0.05,rstride=2, cstride=2)
    #plt.colorbar(surf, shrink=1,aspect=30)
    
    fig.set_size_inches(10,7)
    fig.savefig(name, dpi=100)



def plot_standard_gaussian():
    x1_space = np.linspace(-2,2,40)
    x2_space = np.linspace(-2,2,40)
    

    plot_gaussian(ex1_gaussian, x1_space, x2_space, "Standard isotropic gaussian")

if __name__ == "__main__":
    plot_standard_gaussian()
    print "yes yes yes girl"
