# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 3 exercise2part2.py

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = np.loadtxt('a010_irlsdata.txt')

    labels = data[:,2]
    points = data[:,0:2]
    
    plt.figure(0)
    
    x,y = zip(*points)   
    
    cmap = plt.cm.get_cmap('jet')    
    
    z = plt.scatter(x,y, c=labels, cmap=cmap, alpha=0.7, vmin=0, vmax=1, s=12, edgecolors='none')
    plt.xlabel('$X[:,0]$')
    plt.ylabel('$X[:,1]$')
    plt.colorbar(z)
    
    