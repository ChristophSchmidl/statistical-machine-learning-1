# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 4 exercise3.py
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
from tqdm import tqdm




def plot_hist(X):
    x1, x2, x3, x4 = zip(*X)


    for i,(data,color) in enumerate(zip([x1,x2,x3,x4],['pink','orange','red','white'])):
        i+=1
        #Plot
        fig = plt.figure(0)
        
        linewidth = 0.3        
        if color == 'pink':
            linewidth = 0.1
            
        plt.hist(data, 24, alpha=0.85, facecolor=color,linewidth=linewidth, label=r'$x_{0}$'.format(i))
        #plt.title("Isotropic 2D Gaussian")
        plt.xlabel(r'$value$')
        plt.ylabel(r'$frequency$')
        #plt.colorbar(surf, shrink=1,aspect=30)
        plt.legend(loc='upper right')
        #fig.set_size_inches(8,5)
        fig.savefig("hist_x", dpi=100)    


def scatter_plotta(x,y,colors,xlabel='$x$',ylabel='$Xy$',name=""):
    #cmap = plt.cm.get_cmap('spring')    
    cmap = plt.cm.get_cmap('jet')   
    
    fig = plt.figure(name)
    z = plt.scatter(x,y, c=colors, cmap=cmap, alpha=1, s=12, edgecolors='none')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(z)
    
    fig.set_size_inches(10,7)
    fig.savefig("scatter_"+name, dpi=100)    

if __name__ == "__main__":
    X = np.loadtxt('a011_mixdata.txt')
    x1, x2, x3, x4 = zip(*X)
    #print X
    #plot_hist(X)
    scatter_plotta(x1,x2,x3,"$x_1$","$x_2$","x3")
    scatter_plotta(x1,x2,x4,"$x_1$","$x_2$","x4")
    scatter_plotta(x1,x2,np.array(x3)+np.array(x4),"$x_1$","$x_2$","x3plusx4")