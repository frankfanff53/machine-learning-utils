from .Classifier import Classifier
from numpy.random import default_rng, Generator
import numpy as np


class RandomClassifier(Classifier):

    def __init__(self, rng: Generator = default_rng()) -> None:
        self.rng = rng
        self.catagories = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Fit the training data to the classifier.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.catagories = np.unique(y)

    def predict(self, x: np.ndarray) -> None:
        """ Perform prediction given some examples.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """
        return self.catagories[self.rng.integers(0, self.catagories.size, len(x))]
