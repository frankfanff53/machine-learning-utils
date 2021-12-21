import numpy as np
from .Classifier import Classifier


class NearestNeighbourClassifier(Classifier):
    def __init__(self) -> None:
        self.x = np.array([])
        self.y = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Fit the training data to the classifier.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        """
        self.x = x
        self.y = y

    def predict(self, x: np.ndarray) -> None:
        """ Perform prediction given some examples.

        Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)

        Returns:
        y (np.ndarray): Predicted class labels, numpy array with shape (N,)
        """
        min_elem_indices = np.empty(len(x), dtype=int)
        for i, e in enumerate(x):
            min_index, min_distance = 0, np.inf
            for j, v in enumerate(self.x):
                if j == i:
                    continue
                if np.linalg.norm(e - v) < min_distance:
                    min_index = j
                    min_distance = np.linalg.norm(e - v)
            min_elem_indices[i] = min_index
        return self.y[min_elem_indices]
