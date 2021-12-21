import numpy as np
from .Classifier import Classifier


class WeightedKNNClassifier(Classifier):
    def __init__(self, k: int = 5, w: np.ndarray = None) -> None:
        """ Weighted K-NN Classifier.

        Args:
        k (int): Number of nearest neighbours. Defaults to 5.\
        w (np.ndarray): weights of each neighbor in contribution.
        """
        self.k = k
        self.x = np.array([])
        self.y = np.array([])
        self.w = w

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
            # get k nearest neighbors
            neighbours = np.argpartition(distances, self.k)[:self.k]
            # get weights of neighbors is self.w is None
            weights = self.w
            if weights is None:
                weights = []
                for distance in distances[neighbours]:
                    if distance == 0:
                        weights.append(np.inf)
                    else:
                        weights.append(1 / distance)

            freq_dict = {label: [] for label in set(self.y[neighbours])}
            for (k, label) in enumerate(self.y[neighbours]):
                freq_dict[label].append(k)

            weight_dict = {label: 0 for label in set(self.y[neighbours])}
            for label in freq_dict.keys():
                for index in freq_dict[label]:
                    weight_dict[label] += weights[index]

            result, max_weight = 0, 0
            for label in weight_dict.keys():
                if weight_dict[label] > max_weight:
                    max_weight = weight_dict[label]
                    result = label

            reference_indices[i] = result
        return reference_indices
