import csv
import shutil

import matplotlib.pyplot as plt
import numpy as np

from utils import plot_training


class Perceptron(object):
    """ Classe codant le fonctionnement du perceptron
    dans sa version non stochastique
    """
    def __init__(self, in_features:int, learning_rate:float=0.01, lr_decay:bool=False, max_iter:int=100,
                 early_stopping:bool=False, tol:float=1e-6, display:bool=False) -> None:
        # paramètres générals de la classe
        self.in_features = in_features              # taille d'entrée du perceptron
        self.lr = learning_rate                     # taux d'apprentissage
        self.lr_decay = lr_decay                    # modifie le taux d'apprentissage à chaque itération
                                                    # en le divisant par le nombre d'itération déjà passée
        self.max_iter = max_iter                    # nombre d'epoch
        self.early_stopping = early_stopping        # arrête l'apprentissage si les poids
                                                    # ne s'améliorent pas
        self.tol = tol                              # différence entre avant et après la
                                                    # mise à jour des poids
        self.display = display                      # affichage de l'apprentissage du perceptron

        # initialisation quelconques des connexions synaptiques
        # on considèrera le biais comme la multiplication d'une entrée de valeur 1.0 par un poids associé
        # le biais est utilisé comme seuil de décision du perceptron lors de la prédiction
        self.weights = np.array([-2.0,3.0,6.0])
        #self.weights = np.random.normal(0, 1, size=in_features+1)

    def predict(self, X:np.ndarray) -> float:
        """Prédiction des données d'entrée par le perceptron"""
        result = self.weights[2];
        result +=  np.dot(X,self.weights[0:2])
        if result< 0 :
            return -1
        else :
            return 1
        """
        X est de la frome [nb_data, nb_param]
        La valeur renvoyée est un tableau contenant les prédictions des valeurs de X de la forme [nb_data]
        """


    def fit(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        """Apprentissage du modèle du perceptron

        X : données d'entrée de la forme [nb_data, nb_param]
        y : label associée à X ayant comme valeur
                 1 pour la classe positive
                -1 pour la classe négative
            y est de la forme [nb_data]
        """
        # vérification des labels
        assert np.all(np.unique(y) == np.array([-1, 1]))

        # Sauvegarde tous les calculs de la somme des distances euclidiennes pour l'affichage
        if self.display:
            shutil.rmtree('./img_training', ignore_errors=True)
            metric = []

        # initialisation d'un paramètre permettant de stopper les itérations lors de la convergence
        stabilise = False

        # apprentissage sur les données
        errors = np.zeros(self.max_iter)

        for iteration in range(self.max_iter):
            # variable stockant l'accumulation des coordonnées

            modif_w = np.zeros(len(self.weights))
            erreur = y[0] - y[0]
            erreurpoint=X[0]

            print("---------------")
            print("poids", self.weights)
            for point, label in zip(X, y):
                # prédiction du point
                labelpredict = self.predict(point)

                # accumulation des coordonnées suivant la classe si les données sont mal classées
                if label != labelpredict :
                    errors[iteration] += 1
                    # je prend le premier point mal classé pour le recalcul des poids
                    if errors[iteration]==1 :
                        erreurpoint = point
                        erreur = label - labelpredict



            # affichage de l'erreur et de la ligne séparatrice
            if self.display:
                plot_training(iteration, X, y, self.weights, list(errors[:iteration+1]))

            # mise à jour des poids
            old_weights = np.array(self.weights)
            if self.lr_decay:
                lr = self.lr/(iteration+1)
            else:
                lr = self.lr

            erreurpoint = np.insert(erreurpoint,2,1.)

            self.weights = np.add(old_weights, ((self.lr/(iteration+1)) * erreur * erreurpoint))

            # stopper l'algorithme lorsque l'algorithme converge
            if self.early_stopping:
                # A compléter
                # je pourrais utiliser self.tol et donc soustraire le poids ancien au poids actuel
                # et voir si c'est inférieur ou égale au self.tol pour arrêter
                stabilise = False
                if (self.weights == old_weights).all() :
                    stabilise = True
                if stabilise:
                    # on affiche le dernier hyperplan calculé
                    plot_training(iteration, X, y, self.weights, list(errors[:iteration+1]))
                    # on arrete l'apprentissage
                    break


    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Retourne la moyenne de précision sur les données de test et labels
        """
        return np.sum(y == self.predict(X)) / len(X)
