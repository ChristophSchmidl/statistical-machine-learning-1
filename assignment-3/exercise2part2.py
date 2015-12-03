# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 3 exercise2part2.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
    
#Hessian
def H(phi, y):
    R = np.diag(np.ndarray.flatten(y * (1-y)))
    return np.dot(phi.T,(np.dot(R,phi)))


def scatter_plotta(x,y,colors,xlabel='$X[:,0]$',ylabel='$X[:,1]$'):
    cmap = plt.cm.get_cmap('spring')    
    
    z = plt.scatter(x,y, c=colors, cmap=cmap, alpha=0.7, s=12, edgecolors='none')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(z)

#Gaussian basis function
def gaussian_bf(x, mu):
    sigma = 0.2 * np.identity(2)
    return multivariate_normal.pdf(x,mu,sigma)

def cross_entropy_error(labels,y):
    E = 0
    for label,_y in zip(labels,y):
        E += label * np.log(_y) + (1-label)*np.log(1-_y)
        
    return -E
        

if __name__ == '__main__':

    data = np.loadtxt('a010_irlsdata.txt')

    labels = data[:,2]
    points = data[:,0:2]
    
    plt.figure(0)
    
    x1,x2 = zip(*points)   
    
    scatter_plotta(x1,x2,labels)
    
    #Exercise 2-2-2, 2-2-3
    
    phi = np.array([[1,_x,_y] for _x,_y in zip(x1,x2)])
    t = np.array([[label] for label in labels])
    w = np.array([[0],[0],[0]])
    
    y = sigmoid(np.dot(phi,w))   
    print "ERROR BEFORE:",  cross_entropy_error(labels,y)
    
    for i in range(10):
        y = sigmoid(np.dot(phi,w))
        h = H(phi, y)
        
        w = w - np.dot(np.linalg.inv(h) , (np.dot(phi.T,y-t)))
        print i+1, w
        
        
    plt.figure(1)
    scatter_plotta(x1,x2,y)
    
    print "ERROR:",  cross_entropy_error(labels,y)

    #Exercise 2-2-4, 2-2-5
    
    mu1 = [0,0]
    mu2 = [1,1]
    
    phi1 = gaussian_bf(points,mu1)
    phi2 = gaussian_bf(points,mu2)    
    
    phi = np.array([[1,_x,_y] for _x,_y in zip(phi1,phi2)])
    
    
    plt.figure(2)     
    scatter_plotta(phi1,phi2,labels,xlabel='$\phi_1$',ylabel='$\phi_2$')
    
    t = np.array([[label] for label in labels])
    w = np.array([[0],[0],[0]])
    
    print "GBF"
    
    for i in range(10):
        y = sigmoid(np.dot(phi,w))
        h = H(phi, y)
        
        w = w - np.dot(np.linalg.inv(h) , (np.dot(phi.T,y-t)))
        print i+1, w
        
    
    
    plt.figure(3)
    scatter_plotta(x1,x2,y)    
    
    print "ERROR:",  cross_entropy_error(labels,y)
    
    
    
    