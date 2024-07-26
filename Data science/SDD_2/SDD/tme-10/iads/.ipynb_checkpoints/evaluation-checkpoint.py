# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (np.mean(L),np.std(L)) 
    
def crossval(X, Y, n_iterations, iteration):
    i = int(iteration*(len(X)/n_iterations))
    j = int((iteration+1)*(len(X)/n_iterations)-1)
    Xtest = X[i:j]
    Ytest = Y[i:j]
    Xapp = X[[n for n in range(len(X)) if n not in range(i,j)]]
    Yapp = Y[[n for n in range(len(Y)) if n not in range(i,j)]]
    return Xapp, Yapp, Xtest, Ytest
    
def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    renvoie la liste des performances obtenus, la perfpormance moyenne et l'écart type 
    """
    perf = []
    for i in range(nb_iter) : 
        Xapp,Yapp, Xtest,Ytest = crossval(DS[0],DS[1],nb_iter, i)
        C.train(Xapp,Yapp)
        perf.append(C.accuracy(Xtest,Ytest))
    taux_moyen, taux_ecart = analyse_perfs(perf)
    return (perf,taux_moyen,taux_ecart)
    
