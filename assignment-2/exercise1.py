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


def surf_plot(mu_p, sigma_p):
    
    x,y = np.meshgrid(np.linspace(-0.5,2,250), np.linspace(-0.5,2,250))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    var = multivariate_normal(mean = np.array(mu_p).flatten(), cov = sigma_p)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=35,azim=60)
    
    ax.plot_surface(x,y, var.pdf(pos),cmap=cm.rainbow)

def calculate_prior():
    sigma = np.matrix('0.14 -0.3 0.0 0.2; -0.3 1.16 0.2 -0.8; 0.0 0.2 1.0 1.0; 0.2 -0.8 1.0 2.0')
    labda = np.linalg.inv(sigma)
    mu = np.matrix('1; 0; 1; 2')
    sigma_p = np.linalg.inv(labda[0:2,0:2])
    mu_p = mu[0:2] - sigma_p * labda[0:2,2:4] * (np.matrix('0;0') - mu[2:4])
    
    return mu_p, sigma_p

def generate_data(mu_p, sigma_p):
    
    mu_t = np.random.multivariate_normal(np.array(mu_p).flatten(), sigma_p)
    
    surf_plot(mu_p, sigma_p)
    
    sigma_t = np.matrix([[2.0,0.8],[0.8,4.0]])
    
    data = np.random.multivariate_normal(mu_t, sigma_t, 1000)
    np.savetxt('data.txt',data)
    
    return mu_t, sigma_t

def mu_sigma_maximum_likelihood(data):
    
    
    
    mu_ml = sum(data)/len(data)
    
    sse = [0,0]
    for point in data:
        point = np.matrix(point)        
        sse += (point-mu_ml).T*(point-mu_ml)
    
    sigma_ml =  sse/len(data)

    
    return mu_ml, sigma_ml


def sequential_learning(data):    
    
    N = 0
    mu_ml = None
    
    for point in data:
        
        
    
    pass


if __name__ == "__main__":
    
    np.random.seed(0)
    mu_p, sigma_p = calculate_prior()
    mu_t, sigma_t = generate_data(mu_p, sigma_p)
    data = np.loadtxt('data.txt')
    mu_ml, sigma_ml = mu_sigma_maximum_likelihood(data)
    #sequential_learning(data)
    
    
    #print "True values:\nmu_t:", mu_t, "\nsigma_t: ", sigma_t, "\n------------"
    #print "Maximum likelihood values:\nmu_ml:", mu_ml, "\nsigma_ml: ", sigma_ml, "\n------------"
    #print "Differences:\nmu_t-mu_ml:", mu_t-mu_ml, "\nsigma_t-sigma_ml: ", sigma_t-sigma_ml

    
    
    
    
    
    
    
    
    
    