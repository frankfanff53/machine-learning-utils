import os
import numpy as np
import ml_utils as ml
import pytest

DATASET_PATH = os.getcwd() + '/data/iris.dat'


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_random_classifier(filepath):
    x, y, _ = ml.read_dataset(filepath)
    rng = np.random.default_rng(60012)
    x_train, x_test, y_train, y_test = ml.split_dataset(x, y, 0.2, rng)
    random_classifier = ml.classification.RandomClassifier(rng)
    random_classifier.fit(x_train, y_train)
    random_predictions = random_classifier.predict(x_test)
    assert np.allclose(random_predictions, np.array(
        [0, 1, 1, 0, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 2, 0, 2, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 2, 1, 1]))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
