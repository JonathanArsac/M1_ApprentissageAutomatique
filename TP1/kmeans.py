# author: Benjamin Chamand, Thomas Pelligrini

import shutil
from typing import Tuple

import numpy as np

from utils import plot_training


class KMeans(object):
    def __init__(self, n_clusters:int, max_iter:int, early_stopping:bool=False,
                 tol:float=1e-4, display:bool=False) -> None:
        self.n_clusters = n_clusters            # Nombre de clusters
        self.max_iter = max_iter                # Nombre d'itération
        self.early_stopping =early_stopping    # arrête l'apprentissage si
        self.tol = tol                          # seuil de tolérance entre 2 itérations
        self.display = display                  # affichage des données

        self.cluster_centers = None             # Coordonnées des centres de regroupement
                                                # (centre de gravité des classes)

    def _compute_distance(self, mat:np.ndarray, vec:np.ndarray) -> np.ndarray:
        """Retourne la distance quadratique entre vec1 et vec2 (squared euclidian distance)
        """
        #return np.linalg.norm(vec1-vec2)**2
        # print np.sum((mat-vec)**2,axis=1)

        return np.sum((mat-vec)**2,axis=1)


    def _compute_inertia(self, X:np.ndarray, y:np.ndarray) -> float:
        """Retourne la Sum of Squared Errors entre les points et le centre de leur
        cluster associe
        """
        sse = 0.0
        for label in range(self.n_clusters):
            data = X[y==label] # on recupère dans data tous les pts X ayant le label Y
            sse +=np.sum(self._compute_distance(data, self.cluster_centers[label]))
        return sse


    def _update_centers(self, X:np.ndarray, y:np.ndarray) -> None:
        """Recalcule les coordonnées des centres des clusters
        """

        for label in range(self.n_clusters):
            data = X[y==label]
            self.cluster_centers[label]=np.mean(data,axis=0);


    def predict(self, X:np.ndarray) -> np.ndarray:
        """attribue un indice de cluster à chaque point de data

        X = données
        y = cluster associé à chaque donnée
        """
        data = np.zeros((self.n_clusters,X.shape[0]));
        for label in range(self.n_clusters):
            data[label] = self._compute_distance(X,self.cluster_centers[label])
        y =np.argmin(data,axis=0);
        return y


    def fit(self, X:np.ndarray) -> None:
        """Apprentissage des centroides
        """
        # Récupère le nombre de données
        n_data = X.shape[0]

        # Sauvegarde tous les calculs de la somme des distances euclidiennes pour l'affichage
        if self.display:
            shutil.rmtree('./img_training', ignore_errors=True)
            metric = []

        # 2 cas à traiter :
        #   - soit le nombre de clusters est supérieur ou égale au nombre de données
        #   - soit le nombre de clusters est inférieur au nombre de données
        if self.n_clusters >= n_data:
            # Initialisation des centroides : chacune des données est le centre d'un clusteur
            self.cluster_centers = np.zeros(self.n_clusters, X.shape[1])
            self.cluster_centers[:n_data] = X
        else:
            # Initialisation des centroides

            self.cluster_centers = X[:self.n_clusters,:]

            # initialisation d'un paramètre permettant de stopper les itérations lors de la convergence
            stabilise = False

            # Exécution de l'algorithme sur plusieurs itérations
            for i in range(self.max_iter):
                # détermine le numéro du cluster pour chacune de nos données
                y = self.predict(X)

                # calcule de la somme des distances initialiser le paramètres
                # de la somme des distances
                if i == 0:
                    current_distance = self._compute_inertia(X, y)

                # mise à jour des centroides
                self._update_centers(X, y)

                # mise à jour de la somme des distances
                old_distance = current_distance
                current_distance = self._compute_inertia(X, y)

                # stoppe l'algorithme si la somme des distances quadratiques entre
                # 2 itérations est inférieur au seuil de tolérance

                if self.early_stopping:
                    # A compléter

                    if abs(old_distance-current_distance)<self.tol :
                        stabilise=True

                    if stabilise:
                        diff = abs(old_distance - current_distance)
                        metric.append(diff)
                        plot_training(i, X, y, self.cluster_centers, metric)
                        break

                # affichage des clusters
                if self.display:
                    diff = abs(old_distance - current_distance)
                    metric.append(diff)
                    plot_training(i, X, y, self.cluster_centers, metric)

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Calcule le score de pureté
        """
        n_data = X.shape[0]

        y_pred = self.predict(X)

        score = 0
        for i in range(self.n_clusters):
            _, counts = np.unique(y[y_pred == i], return_counts=True)
            score += counts.max()

        score /= n_data

        return score
