# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2023-2024, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2024

# Import de packages externes
import numpy as np
import pandas as pd
import copy
import math
from iads import arbre as ab

# ---------------------------
        
        
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        correct_predictions = 0
        for i in range(len(desc_set)):
            # Check if the prediction matches the label
            if self.predict(desc_set[i]) == label_set[i]:
                correct_predictions += 1
        return correct_predictions / len(desc_set)
        
        
class ClassifierKNN(Classifier) :
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.k = k  
        self.desc_set = None 
        self.label_set = None
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        dist=np.ndarray(len(self.desc_set))
        
        for i in range(0,len(self.desc_set)):
            dist[i]=np.linalg.norm(x-self.desc_set[i,:])
            
        ind=np.argsort(dist)
        
        
        
        p=0
        for i in range(0,self.k):
            if(self.label_set[ind[i]]==1):
                p += 1
        
        
        if(p/self.k==0.5):
            return 0
        
        else:
            return 2*((p/self.k)-0.5)
    
    
        
    
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        sc=self.score(x)
        if(sc>=0):
            return 1
        else:
            return -1

   

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set=desc_set
        self.label_set=label_set

              
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.uniform(-1, 1, input_dimension)
        self.w /= np.linalg.norm(self.w)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
    
    
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.dimension = input_dimension
        self.epsilon = learning_rate
        self.w = []
        if init==True :
            self.w = np.zeros(self.dimension)
            
        else: 
            for i in range(self.dimension) : 
                v = np.random.uniform(0,1)
                v = 2*v -1 
                v =  v * 0.001
                self.w.append(v)
        self.allw = [self.w.copy()]
                
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        ind = [i for i in range(0,desc_set.shape[0])]
        np.random.shuffle(ind)
        for i in ind : 
            # ŷi = xi * w
            if self.predict(desc_set[i])!= label_set[i] :
                self.allw.append(self.w)
                self.w = self.w + (self.epsilon * label_set[i] *  desc_set[i]) 
                
                
             
            
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        l = []
        for i in range(nb_max) :
            oldw = self.w.copy()
            self.train_step(desc_set,label_set)
            val = np.abs(oldw-self.w)
            norme = np.linalg.norm(val)
            l.append(norme)
            if norme < seuil : 
                break
        return l
            
            
            
            
    def get_allw(self) :
    	return self.allw
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        else :
            return -1 

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        ind = [i for i in range(0,desc_set.shape[0])]
        np.random.shuffle(ind)
        for i in ind : 
            # ŷi = xi * w
            if self.score(desc_set[i])*label_set[i] < 1 :
                self.allw.append(self.w)
                self.w = self.w + self.epsilon * (label_set[i] - self.score(desc_set[i]))*(desc_set[i]) 
        


class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes utilisant la stratégie un-contre-tous avec des classifieurs binaires. """
    def __init__(self, cl_bin):
        """ Constructeur
        Arguments:
            - cl_bin: classifieur binaire à utiliser pour chaque classe.
        """
        self.cl_bin = cl_bin  # Classifieur binaire de base.
        self.classifiers = []  # Liste pour stocker un classifieur pour chaque classe.
        self.classes = []  # Stocker les classes uniques rencontrées lors de l'entraînement.

    def train(self, desc_set, label_set):
        """ Entraînement du modèle sur l'ensemble donné.
        Arguments:
            - desc_set: descriptions des exemples (features).
            - label_set: étiquettes correspondantes.
        """
        self.classes = np.unique(label_set)
        self.classifiers = []
        for c in self.classes:
            # Création d'une nouvelle étiquette pour la stratégie un-contre-tous
            y_tmp = np.where(label_set == c, 1, -1)
            # Clonage du classifieur binaire et entraînement sur les données redéfinies
            clf = copy.deepcopy(self.cl_bin)
            clf.train(desc_set, y_tmp)
            self.classifiers.append(clf)

    def score(self, x):
        """ Calcule les scores de prédiction sur x pour chaque classifieur.
        Arguments:
            - x: une description.
        Retourne:
            - Une liste de scores pour chaque classe.
        """
        scores = [clf.score(x) for clf in self.classifiers]
        return scores

    def predict(self, x):
        """ Renvoie la prédiction sur x.
        Arguments:
            - x: une description.
        Retourne:
            - L'étiquette de classe prédite.
        """
        scores = self.score(x)
        # Sélection de l'indice du score le plus élevé
        index = np.argmax(scores)
        # Retourne l'étiquette de classe correspondante
        return self.classes[index]
        
class ClassifierPerceptronKernel(ClassifierPerceptron):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        ClassifierPerceptron.__init__(self,input_dimension,learning_rate,init)
        self.kernel = noyau
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        V = self.kernel.transform(desc_set)
        ClassifierPerceptron.train_step(self,V,label_set)
        
     
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        V = self.kernel.transform(x)
        ClassifierPerceptron.score(self,V)
        
def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    if len(Y) == 0:
        return None  # Ou retourner une valeur par défaut si approprié

    valeurs, nb_fois = np.unique(Y, return_counts=True)
    # np.argmax retourne l'indice du premier maximum trouvé, ce qui simplifie le code
    ind_max = np.argmax(nb_fois)
    return valeurs[ind_max]
    
def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    tt= np.sum(nb_fois)
    tab = []
    for i in nb_fois : 
        tab.append(i/tt)

    return shannon(tab)
    
def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    somme = 0 
    k = len(P)
    if k > 1 : 
        for i in range(k):
            if P[i] != 0 :
                somme -= (P[i] * math.log(P[i],k))
            
        return (somme)
    
    else : 
        return 0.0

from collections import Counter

class ClassifierKNN_MC(Classifier):
    def __init__(self, input_dimension, k, num_classes):
        super().__init__(input_dimension)
        self.k = k
        self.num_classes = num_classes
        self.desc_set = None
        self.label_set = None
    
    def train(self, desc_set, label_set):
        self.desc_set = np.array(desc_set)
        self.label_set = np.array(label_set)
    
    def predict(self, x):
        
        distances = np.linalg.norm(self.desc_set - x, axis=1)
        # Get the indices of the 'k' nearest neighbors
        nearest_indices = np.argsort(distances)[:self.k]
        
        nearest_labels = self.label_set[nearest_indices]

        
        if isinstance(nearest_labels, np.ndarray):
            nearest_labels = nearest_labels.flatten().tolist()

        
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        return most_common

    
    def score(self, x):
        pass
        
class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = ab.construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def number_leaves(self):
        """ rend le nombre de feuilles de l'arbre
        """
        return self.racine.compte_feuilles()
    
    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------
