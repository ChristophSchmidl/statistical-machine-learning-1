# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 4 exercise2.py
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D



#Gaussian
def ex1_gaussian(x):
    sigma = 0.4 * np.identity(2)

    return 3 * multivariate_normal.pdf(x,[0,0],sigma)

def plot_gaussian(data_function, a_space, b_space, name):
    X, Y = np.meshgrid(a_space, b_space)
        
    z = np.array( [data_function([x,y]) for x,y in zip(np.ravel(X),np.ravel(Y))])
    Z = z.reshape(X.shape)
    
    #Plot
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    #plt.title("Isotropic 2D Gaussian")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    surf = ax.plot_surface(X,Y, Z, cmap=cm.winter, linewidth=0.05,rstride=2, cstride=2)
    #plt.colorbar(surf, shrink=1,aspect=30)
    
    fig.set_size_inches(10,7)
    fig.savefig(name, dpi=100)
    
def plot_gaussian_given(X, Y, Z, name):
        
        
    Z = np.array(Z).reshape(X.shape)
    
    #Plot
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    #plt.title("Isotropic 2D Gaussian")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    surf = ax.plot_surface(X,Y, Z, cmap=cm.winter, linewidth=0.05,rstride=2, cstride=2)
    #plt.colorbar(surf, shrink=1,aspect=30)
    
    fig.set_size_inches(10,7)
    fig.savefig(name, dpi=100)    



def plot_standard_gaussian():
    x1_space = np.linspace(-2,2,40)
    x2_space = np.linspace(-2,2,40)
    
    plot_gaussian(ex1_gaussian, x1_space, x2_space, "std_iso_gaussian")

def activation_tanh(X):
    return np.tanh(np.sum(X))
    
def activation_output(X):
    return np.sum(X)



def forward_pass(layers, X):
    
    X = np.copy(X)
    
    for i, layer in enumerate(layers):
        
        next_x = []
        for node in layer:
            W = node #Weights
            if i == len(layers)-1:
                next_x.append(activation_output(np.dot(X,W)))
            else:
                next_x.append(activation_tanh(np.dot(X,W)))
        X = next_x
    
    return X
    

def create_layer(n_nodes):
    return np.random.rand(n_nodes)-0.5
    
    
def create_layers(n_layers=2, n_input_nodes=2, n_hidden_nodes=8, n_output_nodes=1):
    n_hidden_layers = n_layers - 1
    
    layers = []
    input_layer = create_layer(n_input_nodes)
    layers.append(input_layer)
    
    for n in xrange(n_hidden_layers):
        layer = create_layer(n_hidden_nodes)
        layers.append(layer)
        
    output_layer = create_layer(n_output_nodes)
    layers.append(output_layer)
    
    return layers
    
    

def run_nn():
    x1_space = np.linspace(-2,2,40)
    x2_space = np.linspace(-2,2,40)
    
    x_grid, y_grid = np.meshgrid(x1_space, x2_space)
    
    X = zip(np.ravel(x_grid),np.ravel(y_grid))
    
    #Y = np.array( [ex1_gaussian([x1,x2]) for x1,x2 in x])
    #y = Y.reshape(x_grid.shape)
        
    layers = create_layers()
    
    outputs = []
    for x in X:
        outputs.append(forward_pass(layers, x))
        
    output = outputs
    
  
    plot_gaussian_given(x_grid, y_grid, output, "ex2_2")
    



if __name__ == "__main__":
    np.random.seed(42)
    #plot_standard_gaussian()
    run_nn()
    print "yes yes yes girl"
