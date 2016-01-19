# -*- coding: utf-8 -*-
# Guido Zuidhof s4160703
# Iñez Wijnands s4149696
# SML ASS 4 exercise4.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data():
    N=800
    D=28*28
    X= np.zeros((N,D), dtype=np.uint8)
    
    with open("a012_images.dat", 'rb') as f:
        for n in range(N):
            X[n,:] = np.fromstring(f.read(D), dtype="uint8")
            
    return X

def load_labels():
    N=800
    X= np.zeros(N, dtype=np.uint8)
    
    with open("a012_labels.dat", 'rb') as f:
        for n in range(N):
            X[n] = np.fromstring(f.read(1), dtype="uint8")
            
    return X


def plot_image(im, name=None):
    cmap = plt.cm.get_cmap('spring')
    fig = plt.figure(name)
    plt.imshow(np.reshape(im,(28,28), order='F'),cmap=cmap,interpolation="nearest")
    plt.show()
    
    if not name is None:
        fig.savefig("digits/"+name, dpi=100) 
    

def log_likelihood(data, mu, pi):
    K = len(mu)
    return np.sum(np.log([np.sum([pi[k] * bernoulli(x, mu[k]) for k in range(0, K)]) for x in data]))


def log_likelihood2(X, means, pis):
    X = np.copy(X)
    
    llh = 0
    K = len(means)
    for x in X:
        sk = 0
        for k in range(K):        
            #print pis[k]
            sk+=pis[k] * bernoulli(x, means[k])
        llh += np.log(sk)
        
        
    #print "llh,means,pis", llh, means, pis
    return llh


def likelihood(X, means, pis):
    X = np.copy(X)
    
    lh = 0
    K = len(means)
    for x in X:
        sk = 0
        for k in range(K):        
            #print pis[k]
            sk+=pis[k] * bernoulli(x, means[k])
        lh += sk
        
    return lh

def bernoulli(x, mu):
    return np.product(mu**x * (1-mu)**(1-x))

def plot_confusion_matrix(cm, cmap=plt.cm.spring,K=3):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(K)
    plt.xticks(tick_marks, [str(k+2) for k in range(K)], rotation=45)
    plt.yticks(tick_marks, [str(k+2) for k in range(K)])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def EM(K, X, L, n_iterations=40):
    means = []    
    pis = []
    
    D = X.shape[1]
    N = len(X)
    
    for k in range(K):
        mus = np.random.rand(D)*0.5 + 0.25
        means.append(mus)    
        
    means = np.array([np.ravel(np.random.rand(X.shape[1])*0.5 + 0.25) for _ in range(0, K)])
    
    #True labels
    means = label_means(K,X,L)        
        
    pis = np.repeat(1. / K, K)
    
    for i in range(n_iterations):
        iteration_number = i+1
        
        #E step
        
        print N, K
        gammas = np.zeros((N,K))
        
        for k in range(K):
            gammas[:, k] = [pis[k] * bernoulli(x, means[k]) for x in X]

        gammas = gammas / np.repeat(np.sum(gammas, 1)[np.newaxis], K, axis=0).T

        # Compute sums of gammas
        gamma_sums = np.sum(gammas, 0)

        # Update means
        for k in range(0, K):
            means[k] = 1./gamma_sums[k] * np.dot(gammas[:, k], X)

        # Update pi values
        pis = gamma_sums / np.sum(gamma_sums)

        llh = log_likelihood(X,means,pis)
            
        print "Iteration {0},\t log likelihood {1}".format(iteration_number, llh)        
        
        if (i+1) % 5 == 0 or i < 6:
            for c, m in enumerate(means):
                plot_image(m, name=str(i)+"_k"+str(K)+"_class"+str(c))
    
    classifications = []
    for x in X:
        probabilities = [bernoulli(x, means[k]) for k in range(K)]
        classifications.append(np.argmax(probabilities))
        #classifications[classification].append(x)
    
    
    
    #Prompt user for the labels of found classes
    while(True):
        print "Enter class labels, comma separated"
        labeling = raw_input("> ")
        if "," in labeling:
            labels = map(int, labeling.split(","))
            if len(labels) == K:
                break
        print "Try again"
    
    classifications = [labels[c] for c in classifications]
    
    print accuracy_score(L, classifications)
    conf_matrix = confusion_matrix(L, classifications)
    #Normalize
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plot_confusion_matrix(conf_matrix)
    
   # for class_number, (c, color) in enumerate(zip(classifications, colors)):
        
   #     c1, c2, c3, c4 = zip(*c)  
   #     scatter_one_class(c1,c2,"$x_1$","$x_2$","ex3plots/class_"+str(K)+"_"+str(i+1), color=color, label='class '+str(class_number))        
   #     
    #    print "Class ", class_number, "corrcoef:", np.corrcoef(c1,c2)
   ##     print "Size: ", len(c), "fraction:", len(c)/N
    
    return means, pis


def label_means(K, X, L):
    N = X.shape[1]    
    
    means = np.zeros((3, N))
    
    for label in range(3):
        s = np.zeros(N)
        count = 0
        for i in range(N):
            if L[i] == (label+2):
                s+= X[i]
                count+= 1
                
        means[label] = s/count
    
    return means
    

if __name__ == "__main__":
    X = load_data()
    L = load_labels()
    K = 3
    np.random.seed(1)

    means = label_means(K,X,L)
    
    #Plot real means
    #map(plot_image, means)
      
    EM(K, X, L, n_iterations=40)