# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 3 exercise1.py

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from numpy.random import multivariate_normal

def phi(x):
    return np.array((1,x)).T

def m(x, w=[-0.0445,-0.2021]):
    m_N = np.array(w)

    return np.dot(m_N.T, phi(x))

def s2(x, S_N_inverse):
    return 1/10 + np.dot( phi(x).T, np.dot(np.linalg.inv(S_N_inverse),phi(x)))




if __name__ == '__main__':
    np.random.seed(7)
    m_N = np.array((-0.0445, -0.2021))    
    S_N_inverse = np.array(((22,10),(10,7.2)))
    
    X = np.linspace(0,1,1000)    
    M = np.array([m(x, m_N) for x in X])
    S2 = np.array([s2(x, S_N_inverse) for x in X])
    
    points_x = [0.4,0.6]
    points_t = [0.05,-0.35]
    
    plt.figure(1)
    plt.plot(X, M, color='green')
    
    #Error area
    plt.fill_between(X, M+S2, M-S2, alpha=0.25, color='green')
    plt.scatter(points_x, points_t, s=30, color='black', alpha=0.5)
    plt.xlim(0,1)
    
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    
    W = multivariate_normal(m_N, np.linalg.inv(S_N_inverse),5)
    
    for w in W:
        M = np.array([m(x, w) for x in X])
        plt.plot(X, M, alpha=1, color='pink')
    
    print W
    
    
    