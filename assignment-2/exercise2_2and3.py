# -*- coding: utf-8 -*-
# Iñez Wijnands s4149696 
# Guido Zuidhof s4160703
# SML ASS 2 exercise2_2and3.py

from __future__ import division
import math
import numpy as np
import scipy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def generate_data(a,b,n):
    data = []
    for _ in xrange(n):
        angle = np.random.uniform(0.5*-math.pi, 0.5*math.pi)
        value = b*math.tan(angle)+a
        data.append(value)
    
    return data

def generate_random_position():
    a = np.random.uniform(0.00,10.0)
    b = np.random.uniform(1.0,2.0)
    return a,b



def plot_mean_over_time(data):
    
    
    actual_mean = sum(data)/len(data)
    print "Actual mean:", actual_mean
    
    means = []
    x = xrange(len(data))
    for i in x:
        means.append( sum(data[:i+1])/(i+1))
        
    plt.plot(x, means, label="Running mean")
    plt.plot(x, [actual_mean]*len(data), 'c-',label="Actual mean")
    plt.legend()
    plt.ylabel("Mean")
    plt.xlabel("Number of points used")
    
    
def plot_likelihood(data):
    a_grid,b_grid = np.meshgrid(np.linspace(-10,10,1000), np.linspace(0,5,1000))
    
    vmins = [-20,-30,-40,-240]    
    
    for k,v in zip([1,2,3,20],vmins):
        constant = k * np.log(b_grid/math.pi)
        log_llh = constant
        for x_k in data[:k+1]:
            log_llh = log_llh - np.log((x_k - a_grid)**2 + b_grid**2)
    
        fig = plt.figure(k)
        ax = fig.add_subplot(111, projection='3d')
        plt.title("Log likelihood for k = " + str(k))
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        surf = ax.plot_surface(a_grid,b_grid, log_llh, cmap=cm.winter, vmin=v, vmax=0, linewidth=0.05,rstride=5, cstride=5)
        plt.colorbar(surf, shrink=1,aspect=30)

def log_likelihood(x, data):
    a,b = x
    constant = len(data) * np.log(b/math.pi)
    log_llh = constant
    for x_k in data:
        log_llh = log_llh - np.log((x_k - a)**2 + b**2)
        
    return -log_llh
    
    

def plot_likelihood_function_of_k(data,a,b):
    a_values = []
    b_values = []
    
    k_values = range(len(data))   
    
    for _k in k_values:
        a_guess = 0
        b_guess = 1
        k = _k+1
        [max_a, max_b] = scipy.optimize.fmin(log_likelihood,[a_guess, b_guess], args = (data[:k+1],))
        a_values.append(max_a)
        b_values.append(max_b)
    
    plt.figure(4)
    plt.plot(k_values,a_values, label=r'$\alpha$')
    plt.plot(k_values,b_values, label=r'$\beta$')
    plt.plot(k_values,[a]*len(k_values), label=r'$\alpha_t$')
    plt.plot(k_values,[b]*len(k_values), label=r'$\beta_t$')
    plt.legend(ncol=2)
    plt.xlabel('k')
    plt.ylabel('location (km)')

if __name__ == "__main__":
    np.random.seed(42)
    a,b = generate_random_position()
    print "Lighthouse position: alpha=",a, "beta=",b
    data = generate_data(a,b,200)
    plot_mean_over_time(data)
    #plot_likelihood(data)
    plot_likelihood_function_of_k(data,a,b)