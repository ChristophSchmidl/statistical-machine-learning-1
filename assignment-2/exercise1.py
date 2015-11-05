# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 2 exercise1.py

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal


def surf_plot(mu_p, sigma_p):

    x,y = np.meshgrid(np.linspace(-0.5,2,300), np.linspace(-0.5,2,300))
    
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    var = multivariate_normal(mean = np.array(mu_p).flatten(), cov = sigma_p)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=35,azim=60)

    ax.plot_surface(x,y, var.pdf(pos),cmap=cm.jet, linewidth=0.35,rstride=5, cstride=5)

def calculate_prior():

    sigma = np.matrix('0.14 -0.3 0.0 0.2; -0.3 1.16 0.2 -0.8; 0.0 0.2 1.0 1.0; 0.2 -0.8 1.0 2.0')
    labda = np.linalg.inv(sigma)
    
    mu = np.matrix('1; 0; 1; 2')
    sigma_p = np.linalg.inv(labda[0:2,0:2])
    mu_p = mu[0:2] - sigma_p * labda[0:2,2:4] * (np.matrix('0;0') - mu[2:4])
    
    surf_plot(mu_p, sigma_p)
    
    return mu_p, sigma_p

def generate_data(mu_p, sigma_p):

    mu_t = np.random.multivariate_normal(np.array(mu_p).flatten(), sigma_p)

    sigma_t = np.matrix([[2.0,0.8],[0.8,4.0]])

    data = np.random.multivariate_normal(mu_t, sigma_t, 1000)
    
    #Save data to ascii text file
    np.savetxt('data.txt',data)

    return mu_t, sigma_t

def mu_sigma_maximum_likelihood(data, mu_t, sigma_t):



    mu_ml = sum(data)/len(data)

    sse = [0,0]
    for point in data:
        point = np.matrix(point)
        
        #Our points are not column vectors, that's why the left side
        #is transposed instead of the right
        sse += (point-mu_ml).T*(point-mu_ml)

    sigma_ml =  sse/len(data)
    sigma_ml_unbiased = sse * (1/(len(data)-1))


    print "True values:\nmu_t:", mu_t, "\nsigma_t: ", sigma_t, "\n------------"
    print "Maximum likelihood values:\nmu_ml:", mu_ml, "\nsigma_ml: ", sigma_ml
    print "\nsigma_ml_unbiased: ", sigma_ml_unbiased, "\n------------"
    
    print "Differences:\nmu_t-mu_ml:", mu_t-mu_ml, "\nsigma_t-sigma_ml: ", sigma_t-sigma_ml, "\n------------"
    print "Sigma unbiased: \nsigma_t-sigma_ml_unbiased: ", sigma_t-sigma_ml_unbiased, "\n------------"

    return mu_ml, sigma_ml


def sequential_learning_ml(data):

    N = 0
    mu_ml = 0

    mus = []

    for point in data:
        N += 1
        mu_ml = mu_ml + (1/N)*(point-mu_ml)
        mus.append(mu_ml)

    print "Sequential mu_ml:", mu_ml
    return mus

def sequential_learning_map(data, mu_p, sigma_p, sigma_t):

    sigma = sigma_p
    mu = mu_p

    mus = []

    for point in data:
        #Our points are not column vectors, now it is.
        point = np.matrix(point).T
        
        S =  np.linalg.inv( np.linalg.inv(sigma) + np.linalg.inv(sigma_t))
        mu = np.dot(S, np.dot( np.linalg.inv(sigma_t), point) + np.dot( np.linalg.inv(sigma),  mu))
        sigma = S

        mus.append(np.array(mu))

    print "Sequential mu_map:", mu
    return mus


#Plot the various mu's (estimated by ML, MAP and true)
def plot_mu(mu_t, mu_mls, mu_maps):
    plt.figure(1)
    plt.plot(range(1000),zip(*mu_mls)[0], label="ML for $\mu_{t1}$") 
    plt.plot(range(1000),zip(*mu_maps)[0], label="MAP for $\mu_{t1}$") 
    plt.plot(range(1000),[mu_t[0]]*1000, label="$\mu_{t1}$") 
    plt.plot(range(1000),zip(*mu_mls)[1], label="ML for $\mu_{t2}$") 
    plt.plot(range(1000),zip(*mu_maps)[1], label="MAP for $\mu_{t2}$") 
    plt.plot(range(1000),[mu_t[1]]*1000, label="$\mu_{t2}$") 
    plt.legend(ncol=2)


if __name__ == "__main__":

    np.random.seed(0)
    mu_p, sigma_p = calculate_prior()
    mu_t, sigma_t = generate_data(mu_p, sigma_p)
    
    #Load data from text file
    data = np.loadtxt('data.txt')
    mu_ml, sigma_ml = mu_sigma_maximum_likelihood(data, mu_t, sigma_t)

    mu_mls = sequential_learning_ml(data)
    mu_maps = sequential_learning_map(data, mu_p, sigma_p, sigma_t)
    
    plot_mu(mu_t,mu_mls,mu_maps)











