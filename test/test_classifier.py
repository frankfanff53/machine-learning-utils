import os
import pytest
import numpy as np
import ml_utils as ml
from ml_utils import classification

DATASET_PATH = os.getcwd() + '/data/iris.dat'
x, y, _ = ml.read_dataset(DATASET_PATH)
rng = np.random.default_rng(60012)
x_train, x_test, y_train, y_test = ml.split_dataset(x, y, 0.2, rng)


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_random_classifier(filepath):
    random_classifier = classification.RandomClassifier(rng)
    random_classifier.fit(x_train, y_train)
    random_predictions = random_classifier.predict(x_test)
    assert np.allclose(random_predictions, np.array(
        [0, 1, 1, 0, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 2,
            0, 2, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 1]
    ))


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_nn_classifier(filepath):
    nn_classifier = classification.NearestNeighbourClassifier()
    nn_classifier.fit(x_train, y_train)
    nn_predictions = nn_classifier.predict(x_test)
    assert np.allclose(nn_predictions, np.array(
        [1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 0, 2,
            0, 0, 2, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ))


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_knn_classifier(filepath):
    knn_classifier = classification.KNNClassifier()
    knn_classifier.fit(x_train, y_train)
    knn_predictions = knn_classifier.predict(x_test)
    assert np.allclose(knn_predictions, np.array(
        [1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 0, 2,
            0, 0, 2, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ))


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_weighted_knn_classifier(filepath):
    wknn_classifier = classification.WeightedKNNClassifier()
    wknn_classifier.fit(x_train, y_train)
    wknn_predictions = wknn_classifier.predict(x_test)
    assert np.allclose(wknn_predictions, np.array(
        [1, 2, 1, 2, 0, 1, 2, 0, 0, 2, 2, 2, 2, 0, 2,
            0, 0, 2, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
