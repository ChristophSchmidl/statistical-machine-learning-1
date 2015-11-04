# -*- coding: utf-8 -*-
# Iñez Wijnands s4149696 
# Guido Zuidhof s4160703
# SML ASS 2 exercise2.py

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as plt


def distr():
    return [b / (math.pi*((b**2)+((_x-a)**2))) for _x in x]


def plot_distribution():
    a = 0.5
    b = 1.0
    
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, distr(a,b,x), 'b-', lw=1.5)
    plt.xlabel (r'$x_k$')
    plt.ylabel(r'$p(x_k|\alpha,\beta)$')
    


def log_posterior_density_plot():
    data = [4.8, -2.7, 2.2, 1.1, 0.8, -7.3]
    
    a_values = np.linspace(-5,-5,1000)
    
    
    #for a in a_values:

           # np.product()
        
    
    
    pass
    
    
    


if __name__ == "__main__":
    plot_distribution()