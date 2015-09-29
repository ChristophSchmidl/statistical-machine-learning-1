# -*- coding: utf-8 -*-
# Inez Wijnando s4149696 
# Guido Zuidhof s4160703
# SML ASS 1
from __future__ import division
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm



def h(x,y):
    return 100 * ((y-(x**2))**2) + (1-x)**2

#Derivative x
def h_dx(x,y):
    return - 400 * (x*y) + 400*(x**3) + 2*x - 2
    
def h_dy(x,y):
    return 200*y - 200 * (x**2)
    
def sgd(x=-2,y=2, eta=0.001, max_iteration=500000):
    
    iteration = 0
    
    x_coords = [x]
    y_coords = [y]    
    
    while iteration < max_iteration :
        step_x = - eta * h_dx(x,y)   
        step_y = - eta * h_dy(x,y)
        
        x = x + step_x
        y = y + step_y
        
        #print iteration, "X:",x, "Y:",y, "\th(x,y):", h(x,y)
        iteration += 1
        
        x_coords.append(x)
        y_coords.append(y)
        
        
        
        if np.abs(step_x) < eta*(10**-5) and np.abs(step_y) < eta*(10**-5):
            break
    
    print iteration, "X:",x, "Y:",y, "\th(x,y):", h(x,y)
    return x_coords, y_coords, iteration
    


if __name__ == "__main__":
    
    
    etas = [0.001, 0.0001, 0.00001]    
    
    for eta in etas:
    
        plt.close()
        fig = plt.figure()
        
        ax = fig.gca(projection='3d')
        
        X = np.arange(-2,2,0.03)
        Y = np.arange(-1.0, 3.0, 0.03)
        X, Y = np.meshgrid(X, Y)
        
        z = np.array( [h(x,y) for x,y in zip(np.ravel(X),np.ravel(Y))])
        Z = z.reshape(X.shape)
        
        ax.plot_surface(X, Y, Z,cmap=cm.rainbow)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('H(x,y)')
        
        ax.view_init(elev=90, azim=50)
        
        
        max_iter = 10000000
        x_coords, y_coords, iterations = sgd(eta=eta, max_iteration = max_iter)
        
        converged = 'n' if max_iter == iterations else 'y'
        
        
        z_coords = [h(x,y) for x,y in zip(x_coords,y_coords)]
        ax.plot(x_coords, y_coords, z_coords,  color='#FFFF00')    
        
        plt.savefig('trajectory{0}{1}{2}.png'.format(eta,converged,iterations))
        plt.show()
    
