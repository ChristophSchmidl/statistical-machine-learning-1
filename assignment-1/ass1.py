# -*- coding: utf-8 -*-
# Inez Wijnando s4149696 
# Guido Zuidhof s4160703
# SML ASS 1
import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 + math.sin(8*x + 1)

def createNoise(n):
    return np.random.normal(0,0.1,n)

def createData(n=10):
    X = np.linspace(0,1,num = n)
    Y = map(f, X)
    noise = createNoise(n)
    Y+=noise
    D = np.vstack((X,Y))
    return D
    
def PolCurFit(D,m):
    pass


if __name__ == "__main__":
    D = createData()
    
    
    plt.close()
    plt.plot(D[0], D[1], 'b+')
    
    plt.plot(np.linspace(0,1,num = 1000),[f(x) for x in np.linspace(0,1,num = 1000)], 'r')
    plt.show()
    
    T = createData(100)
    
    