# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 3 exercise2.py

import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def hessian_sin(x):
    return x + (np.cos(x)/np.sin(x))

def run_hessian_sin(x, n_iter=5):
    print "x_0:", x
    
    for i in xrange(n_iter):
        x = hessian_sin(x)
        print 'x_'+str(i+1)+":",x
    
def H(phi, y):
    R = np.diag(np.ndarray.flatten(y * (1-y)))
    return np.dot(phi.T,(np.dot(R,phi)))


if __name__ == '__main__':
    
    run_hessian_sin(1)
    run_hessian_sin(-1)
        
    phi = np.array([[1,0.3],[1,0.44],[1,0.46],[1,0.6]])
    t = np.array([[1],[0],[1],[0]])
    w = np.array([[1],[1]])
    
    for i in range(10):
        y = sigmoid(np.dot(phi,w))
        h = H(phi, y)
        
        w = w - np.dot(np.linalg.inv(h) , (np.dot(phi.T,y-t)))
        print i+1, w
    
    
        
    