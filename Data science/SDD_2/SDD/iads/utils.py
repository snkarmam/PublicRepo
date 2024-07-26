# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""


# Fonctions utiles
# Version de départ : Février 2024

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    # ensemble des données 
    desc_pos = desc_set[label_set == +1]
    desc_neg = desc_set[label_set == -1]
    # tirer n- et n+ exemplaires dans desc neg et pos 
    train_ind = random.sample([i for i in range(0,desc_set.shape[0])], n_pos + n_neg)
    
    
    # Xtrain et Ytrain
    xtrain = desc_set[train_ind]
    ytrain = label_set[train_ind]
    
    #le reste Xtest et Ytest
    ind_rest = [ j for j in range(0,desc_set.shape[0]) if j not in train_ind]
    
    x_test = desc_set[ind_rest]
    y_test = label_set[ind_rest]
    
    return (xtrain,ytrain),(x_test,y_test)
    
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    return (np.random.uniform(binf,bsup,(n*2,p)),np.asarray([-1 for i in range(0,n)]+[+1 for i in range(0,n)]))

# genere_dataset_gaussian:

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """

    # COMPLETER ICI (remplacer la ligne suivante)    
    pos = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    neg = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    label = np.concatenate((np.ones(nb_points) * -1, np.ones(nb_points))).astype(int)
    return np.concatenate((neg, pos)), label

# plot2DSet:

def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
        desc -> les données
        labels -> ens des etiquettes 
    """
    negatifs = desc[labels == -1]
    positifs = desc[labels == +1]
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o', color="red")
    plt.scatter(positifs[:,0],positifs[:,1],marker='x', color="blue")
    daccord 
    
# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
    
    
    
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    v = np.array([[0+var,0],[0,0+var]])
    neg0 = np.random.multivariate_normal([0, 0],v, n)
    neg1 = np.random.multivariate_normal([1, 1],v,n)
    pos0 = np.random.multivariate_normal([1, 0], v, n)
    pos1 = np.random.multivariate_normal([0, 1], v,n )
    label = np.concatenate((np.ones(n*2)* -1 , np.ones(n*2))).astype(int)
    
    return np.concatenate((neg0,neg1,pos0,pos1)),label
    


