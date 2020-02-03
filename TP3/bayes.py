import numpy as np
from typing import Union


class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.empty(n_obs)
        
        ...
        
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        self.sigma = np.zeros((n_classes, n_features, n_features))
        # learning
        ...

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
