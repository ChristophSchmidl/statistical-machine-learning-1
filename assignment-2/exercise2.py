# -*- coding: utf-8 -*-
# Iñez Wijnands s4149696 
# Guido Zuidhof s4160703
# SML ASS 2 exercise2.py

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as plt


def distr_single(x,a,b):
    return [b / (math.pi*(b**2+((_x-a)**2))) for _x in x]

def distr_data(data,a,b):
    return np.product([b / math.pi * (1/ ((d-a)**2 + b**2)) for d in data])


def plot_distribution():
    a = 0.5
    b = 1.0
    
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, distr(x,a,b), 'g-', lw=1.5)
    plt.xlabel (r'$x_k$')
    plt.ylabel(r'$p(x_k|\alpha,\beta)$')
    


def plot_with_data():
    data = [4.8, -2.7, 2.2, 1.1, 0.8, -7.3]
    b = 1
    
    a_values = np.linspace(-5,5,1000)
    y_values = [distr_data(data,a, b) for a in a_values]
    
    
    plt.plot(a_values, y_values, 'g-', lw=1.5) 
    plt.xlabel (r'$\alpha$')
    plt.ylabel(r'$p(\alpha|\mathcal{D},\beta=1)$')
    plt.xlim(-5,5)
    


if __name__ == "__main__":
    #plot_distribution()
    plot_with_data()