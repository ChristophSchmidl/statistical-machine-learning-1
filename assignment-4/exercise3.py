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
    fig.savefig(name, dpi=100)    
    
    
def scatter_one_class(x,y,xlabel='$x$',ylabel='$Xy$',name="", color='green', label='data'):   
    fig = plt.figure(name)
    plt.scatter(x,y, alpha=1, s=12, color=color, edgecolors='none', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    
    fig.set_size_inches(10,7)
    fig.savefig(name, dpi=100) 


def log_likelihood(X, means, covariances, pis):
    llh = 0
    K = len(means)
    for x in X:
        sk = 0
        for k in range(K):        
            #print pis[k]
            sk+=pis[k] * multivariate_normal.pdf(x, means[k], covariances[k])
        llh += np.log(sk)
        
    return llh


def EM(K, x1, x2, x3, x4):
    X = zip(x1,x2,x3,x4)
    X = np.array(X)
    means = []    
    covariances = []
    pis = []
    
    for k in range(K):
        mus = []
        for x in [x1,x2,x3,x4]:
            x_mean = np.mean(x)
            epsilon = np.random.random()*2 - 1 
            mus.append(x_mean + epsilon)
            
        means.append(mus)
        sigma = np.identity(X.shape[1]) * 4 * np.random.random() + 2
        covariances.append(sigma)
        
        pis.append(1)
    
    for i in range(100):
        iteration_number = i+1
        
        #E step
        
        pi_probs = np.zeros((K, len(X)))
        gammas = np.zeros((K,len(X)))
        gamma_sums = np.zeros((K)) #N_k
        
        for k in range(K):
            pi_probs[k] = pis[k] * multivariate_normal.pdf(X, means[k], covariances[k])
            
        
        pi_probs_sum = sum(pi_probs)
        
        for k in range(K):
            gammas[k] = pi_probs[k]/pi_probs_sum
            gamma_sums[k] = sum(gammas[k]) 
        
        
        N = len(x)
        D = X.shape[1]
        #M step
        
        for k in range(K):          
            
            sum_gamma_x = np.zeros(D)
            for n in xrange(N):
                sum_gamma_x += gammas[k,n] * X[n] 
                
            means[k] = (1/gamma_sums[k]) * sum_gamma_x
            
            sum_gamma_dist = np.zeros((D,D))
            
            for n in xrange(N):
                sum_gamma_dist += gammas[k,n] * ( np.multiply(np.matrix(X[n]-means[k]), np.matrix(X[n]-means[k]).T))
            
            
            covariances[k] = (1/gamma_sums[k]) * sum_gamma_dist
            pis[k] = gamma_sums[k] / N
            
        llh = log_likelihood(X,means,covariances,pis)
            
        print "Iteration {0},\t log likelihood {1}".format(iteration_number, llh)        
        
        
    classifications = [[] for k in range(K)] #
    for x in X:
        probabilities = [multivariate_normal.pdf(x, means[k], covariances[k]) for k in range(K)]
        classification = np.argmax(probabilities)
        classifications[classification].append(x)
        
        
    cmap = plt.cm.get_cmap('jet')
    
    colors = [ cmap(k/(K-1)) for k in range(K)]
    
        
    for class_number, (c, color) in enumerate(zip(classifications, colors)):
        
        c1, c2, c3, c4 = zip(*c)  
        scatter_one_class(c1,c2,"$x_1$","$x_2$","ex3plots/class_"+str(K)+"_"+str(i+1), color=color, label='class '+str(class_number))        
        
        print "Class ", class_number, "corrcoef:", np.corrcoef(c1,c2)
                
        

if __name__ == "__main__":
    X = np.loadtxt('a011_mixdata.txt')
    x1, x2, x3, x4 = zip(*X)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    
    EM(2, x1, x2, x3, x4)
    
    #print X
    #plot_hist(X)
    #scatter_plotta(x1,x2,x3,"$x_1$","$x_2$","x3")
    #scatter_plotta(x1,x2,x4,"$x_1$","$x_2$","x4")
    #scatter_plotta(x1,x2,np.array(x3)+np.array(x4),"$x_1$","$x_2$","x3plusx4")