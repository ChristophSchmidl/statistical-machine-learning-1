# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 4 exercise4.py
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    N=800
    D=28*28
    X= np.zeros((N,D), dtype=np.uint8)
    
    with open("a012_images.dat", 'rb') as f:
        for n in range(N):
            X[n,:] = np.fromstring(f.read(D), dtype="uint8")
            
    return X


if __name__ == "__main__":
    X = load_data()
    cmap = plt.cm.get_cmap('spring')
    for x in X[:100]:
        plt.imshow(np.reshape(x,(28,28), order='F'),cmap=cmap,interpolation="nearest")
        plt.show()