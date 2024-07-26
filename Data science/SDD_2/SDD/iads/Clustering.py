# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from collections import OrderedDict
# ------------------------ 

def normalisation(dataframe):
    # Copier le dataframe pour éviter de modifier l'original
    df_normalise = dataframe.copy()
    # Boucler sur chaque colonne pour normaliser
    for colonne in df_normalise.columns:
        min_col = df_normalise[colonne].min()
        max_col = df_normalise[colonne].max()
        df_normalise[colonne] = (df_normalise[colonne] - min_col) / (max_col - min_col)
    return df_normalise

def dist_euclidienne(vecteur1, vecteur2):
    # Convertir les DataFrame en numpy array si nécessaire
    if isinstance(vecteur1, pd.DataFrame):
        vecteur1 = vecteur1.to_numpy()
    if isinstance(vecteur2, pd.DataFrame):
        vecteur2 = vecteur2.to_numpy()
    
    # Calculer la distance euclidienne
    distance = np.sqrt(np.sum((vecteur1 - vecteur2) ** 2))
    return distance

def centroide(donnees):
    # Convertir le DataFrame en numpy array si nécessaire
    if isinstance(donnees, pd.DataFrame):
        donnees = donnees.to_numpy()
    
    # Calculer le centroïde
    centroide = np.mean(donnees, axis=0)
    return centroide

def dist_centroides(groupe1, groupe2):
    # Calculer les centroïdes des deux groupes
    centroide1 = centroide(groupe1)
    centroide2 = centroide(groupe2)
    # Calculer et renvoyer la distance euclidienne entre les deux centroïdes
    return dist_euclidienne(centroide1, centroide2)

def initialise_CHA(df):
    partition = {i: [i] for i in df.index}
    return partition

def fusionne(dataframe,P0,verbose= False) : 
    P1 = P0.copy() 
        
    dm = 1e10
    ind1 = -1
    ind2 = -1
    for i in P0 :
        for j in P0 :
            if i != j : 
                if dm > dist_centroides(dataframe.iloc[P0[i]],dataframe.iloc[P0[j]]) : 
                    dm = dist_centroides(dataframe.iloc[P0[i]],dataframe.iloc[P0[j]])
                    ind1 = i 
                    ind2 = j
    P1[max(P0.keys())+1]  = P0[ind1]+ P0[ind2]
    del P1[ind1]
    del P1[ind2]

    if verbose == True : 
        print("fusionne: distance mininimale trouvée entre",[ind1,ind2],"= ",dm)
        print("fusionne: les 2 clusters dont les clés sont ",[ind1,ind2], " sont fusionnés")
        print("fusionne: on crée la  nouvelle clé ",len(P0.keys()),"  dans le dictionnaire.")
        print("fusionne: les clés de  ",[ind1,ind2],"  sont supprimées car leurs clusters ont été fusionnés.")
    return (P1,ind1,ind2,dm)

def CHA_centroid(dataframe,verbose =False,dendrogramme=False) : 
    derniere_valeur = 0
    partition = initialise_CHA(dataframe)
    dct = partition.copy()
    l = []
    
    while len(dct.keys()) != 1 : 
        e1,ind1,ind2,dm = fusionne(dataframe,dct,verbose)
        dct = e1
        derniere_cle = list(dct.keys())[-1]
        derniere_valeur = dct[derniere_cle]
        l.append([ind1,ind2,dm,len(derniere_valeur)])

    if dendrogramme == True : 
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            CHA_centroid(dataframe), 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return l 


def dist_complete(c1,c2):
    maxc = -1e100
    ind1 = -1 
    ind2 = -1 
    for i,x in enumerate(np.array(c1)) : 
        for j,y in  enumerate(np.array(c2)):
            if maxc < dist_euclidienne(x,y) : 
                maxc = dist_euclidienne(x,y)
                ind1 = i
                ind2 = j
    return (maxc,(ind2,ind1))

def dist_simple(c1,c2):
    mins = 1e100
    ind1 = -1 
    ind2 = -1 
    for i,x in enumerate(np.array(c1)) : 
        for j,y in  enumerate(np.array(c2)):
            if mins > dist_euclidienne(x,y) : 
                mins = dist_euclidienne(x,y)
                ind1 = i
                ind2 = j
    return (mins,(ind2,ind1))


def dist_average(c1,c2):
    lc1 = len(c1) 
    lc2 = len(c2)
    somme = 0
    
    for i,x in enumerate(np.array(c1)) : 
        for j,y in  enumerate(np.array(c2)):
            somme += dist_euclidienne(x,y)
    return ((1/lc1)*(1/lc2)*(somme),(lc1*lc2))

def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    cent = centroide(Ens)
    somme = 0
    for elem in np.array(Ens) :
        somme += (dist_euclidienne(elem,cent))**2 
    return somme 

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """
    lens = np.array(Ens)
    ind = [i for i in range(0,len(lens))]
    np.random.shuffle(ind)
    l = ind[:K]
    
    return lens[l] 

def plus_proche(Exe, Centres):
    """ Array * Array -> int
        Exe : Array contenant un exemple
        Centres : Array contenant les K centres
    """
    min_dist = float('inf')  # Initialise avec l'infini
    indice_plus_proche = -1

    for j, centre in enumerate(Centres):
        dist = dist_euclidienne(Exe, centre)  # Calcul de la distance
        if dist < min_dist: 
            min_dist = dist
            indice_plus_proche = j
    return indice_plus_proche


def affecte_cluster(Base,Centres):
    """ Array * Array -> dict[int,list[int]]
        Base: Array contenant la base d'apprentissage
        Centres : Array contenant des centroides
    """
    dic = dict()
    for i in range(0,len(Base)):
        c = plus_proche(Base.iloc[i], Centres)
        if c not in dic.keys() : 
            dic[c] = []
        dic[c].append(i)


    return dict(OrderedDict(sorted(dic.items())))


def nouveaux_centroides(Base,U):
    """ Array * dict[int,list[int]] -> DataFrame
        Base : Array contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    l = []
    for i in U.keys(): 
        l.append(list(centroide(Base.iloc[U[i]])))
    return np.array(l)
        
def inertie_globale(Base, U):
    """ Array * dict[int,list[int]] -> float
        Base : Array pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    somme = 0
    for i in U.keys():
        somme += inertie_cluster(Base.iloc[U[i]])
    return somme

def kmoyennes(K, Base, epsilon, iter_max):
    """
    int * Array * float * int -> tuple(Array, dict[int,list[int]])
    K : entier > 1 (nombre de clusters)
    Base : Array pour la base d'apprentissage
    epsilon : réel > 0
    iter_max : entier > 1
    """
    # Définir des centroides
    Centres = init_kmeans(K, Base)
    
    for i in range(iter_max):
        # définir matrice d'affectation
        Affect = affecte_cluster(Base, Centres)
        new_Centres = np.array([centroide(Base.iloc[Affect[k]]) for k in range(K)] )
        # Calcul de l'inertie globale actuelle
        int_glo = inertie_globale(Base, Affect)
        # Critère de convergence
        if i > 0 and np.abs(int_glo - iner_pre) < epsilon:
            print("itération ",i," Inertie : ",int_glo," Difference: ",np.abs(int_glo - iner_pre))
            return (new_Centres, Affect)
    
        if i == 0 : 
            print("itération ",i," Inertie : ",int_glo," Difference: ",np.abs(int_glo))
        if i > 0 :
            print("itération ",i," Inertie : ",int_glo," Difference: ",np.abs(int_glo - iner_pre))
                  
        Centres = new_Centres
        iner_pre = int_glo
        
    return (Centres, Affect)