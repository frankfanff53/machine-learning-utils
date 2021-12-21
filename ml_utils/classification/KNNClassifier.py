import numpy as np
from .Classifier import Classifier


class KNNClassifier(Classifier):
    def __init__(self, k=5) -> None:
        """ K-NN Classifier.

        Args:
        k (int): Number of nearest neighbours. Defaults to 5.
        """
        self.k = k
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x, y):
        """ Fit the training data to the classifier.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """ Perform prediction given some examples.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """

        reference_indices = np.empty(len(x), dtype=int)
        for i, e in enumerate(x):
            distances = np.empty(len(self.x))
            for j, v in enumerate(self.x):
                if j == i:
                    distances[j] = np.inf
                else:
                    distances[j] = np.linalg.norm(e - v)
            neighbours = np.argpartition(distances, self.k)[:self.k]
            labels, freq = np.unique(self.y[neighbours], return_counts=True)
            reference_indices[i] = labels[np.argmax(freq)]
        return reference_indices
