import os
import pytest
import numpy as np
import ml_utils as ml
from ml_utils import classification, evaluation

DATASET_PATH = os.getcwd() + '/data/iris.dat'
x, y, _ = ml.read_dataset(DATASET_PATH)
rng = np.random.default_rng(60012)
x_train, x_test, y_train, y_test = ml.split_dataset(x, y, 0.2, rng)

random_classifier = classification.RandomClassifier(rng)
random_classifier.fit(x_train, y_train)
random_predictions = random_classifier.predict(x_test)
random_confusion = evaluation.confusion_matrix_init(y_test, random_predictions)

knn_classifier = classification.KNNClassifier(k=1)
knn_classifier.fit(x_train, y_train)
knn_predictions = knn_classifier.predict(x_test)
knn_confusion = evaluation.confusion_matrix_init(y_test, knn_predictions)


@pytest.mark.parametrize('m, n', [(3, 20)])
def test_k_fold_split(m, n):
    a, b, c = evaluation.k_fold_split(m, n, rng)
    assert np.allclose(a, np.array([1, 0, 13, 11, 17, 18, 16]))
    assert np.allclose(b, np.array([2, 3, 15, 10, 9, 5, 6]))
    assert np.allclose(c, np.array([19, 8, 14, 4, 7, 12]))
