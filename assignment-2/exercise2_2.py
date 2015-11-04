# -*- coding: utf-8 -*-
# Iñez Wijnands s4149696 
# Guido Zuidhof s4160703
# SML ASS 2 exercise2_2.py

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as plt

ANGLE_BOUNDS = (-math.pi/2, math.pi/2)


def generate_random_position():
    a = np.random.uniform(0.00,10.0)
    b = np.random.uniform(1.0,2.0)
    return a,b

def generate_data(a,b,n):
    data = []
    for _ in xrange(n):
        angle = np.random.uniform(ANGLE_BOUNDS[0], ANGLE_BOUNDS[1])
        value = b*math.tan(angle)+a
        data.append(value)
    
    
    return data

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
    

if __name__ == "__main__":
    np.random.seed(42)
    a,b = generate_random_position()
    print "Lighthouse position: alpha=",a, "beta=",b
    data = generate_data(a,b,200)
    plot_mean_over_time(data)
    