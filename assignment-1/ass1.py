# -*- coding: utf-8 -*-
# Inez Wijnando s4149696 
# Guido Zuidhof s4160703
# SML ASS 1
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def f(x):
    return 1 + math.sin(8*x + 1)

def createNoise(n):
    return np.random.normal(0,0.3,n)

def createData(n=10):
    X = np.linspace(0,1,num = n)
    Y = map(f, X)
    noise = createNoise(n)
    Y+=noise
    D = np.vstack((X,Y))
    return D
    
def PolCurFit(D,M):
    
    x = D[0]
    t = D[1]
    N = len(x)
    
    def sum_sq_error(w):
        _sum = 0
        
        for n in xrange(N):
            _sum +=  (poly_y(x[n],w)-t[n])**2
            
        _sum = 0.5 * _sum
        return _sum
    
    w0 = np.zeros((M+1,))
        
    
    result = scipy.optimize.minimize(sum_sq_error, w0)
    
    return result.x, result.fun
    

def poly_y(x,w):
    y = 0.0
    
    for i in xrange(len(w)):
        y += w[i]* (x**i)
        
    return y
    
    



if __name__ == "__main__":
    np.random.seed(1337)
    D = createData()
    
    plt.close()
    plt.plot(D[0], D[1], 'r+')
    
    plt.plot(np.linspace(0,1,num = 1000),[f(x) for x in np.linspace(0,1,num = 1000)], 'b')
    
    
    T = createData(100)
    
 
    
    weights, err = PolCurFit(D,9)
    print weights, "\nError: ",err
    
    plt.plot(np.linspace(0,1,num = 1000),[poly_y(x,weights) for x in np.linspace(0,1,num = 1000)], 'g')
    plt.show()
    
    
    