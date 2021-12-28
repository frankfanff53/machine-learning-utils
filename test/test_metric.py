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


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_confustion_matrix(filepath):
    assert np.allclose(
        evaluation.confusion_matrix_init(y_test, random_predictions), np.array(
            [4, 3, 4, 5, 2, 1, 4, 4, 3]
        ).reshape((3, 3)))

    assert np.allclose(
        evaluation.confusion_matrix_init(y_test, knn_predictions), np.array(
            [11, 0, 0, 0, 8, 0, 0, 1, 10]
        ).reshape((3, 3)))


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_accuracy(filepath):
    assert np.isclose(
        evaluation.accuracy(y_test, random_predictions), 0.3)
    assert np.isclose(
        evaluation.accuracy(y_test, knn_predictions), 0.9666666666666667)


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_accuracy_from_confusion(filepath):
    assert np.isclose(
        evaluation.accuracy_from_confusion(random_confusion), 0.3)
    assert np.isclose(
        evaluation.accuracy_from_confusion(knn_confusion), 0.9666666666666667)


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_precision(filepath):
    (p_random, macro_p_random) = evaluation.precision(y_test, random_predictions)
    (p_knn, macro_p_knn) = evaluation.precision(y_test, knn_predictions)

    assert np.allclose(p_random, np.array([0.30769231, 0.22222222, 0.375]))
    assert np.isclose(macro_p_random, 0.30163817663817666)

    assert np.allclose(p_knn, np.array([1., 0.88888889, 1.]))
    assert np.isclose(macro_p_knn, 0.9629629629629629)


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_recall(filepath):
    (r_random, macro_r_random) = evaluation.recall(y_test, random_predictions)
    (r_knn, macro_r_knn) = evaluation.recall(y_test, knn_predictions)

    assert np.allclose(r_random, np.array([0.36363636, 0.25, 0.27272727]))
    assert np.isclose(macro_r_random, 0.29545454545454547)

    assert np.allclose(r_knn, np.array([1., 1., 0.90909091]))
    assert np.isclose(macro_r_knn, 0.9696969696969697)


@pytest.mark.parametrize('filepath', [DATASET_PATH])
def test_f1_score(filepath):
    (f_random, macro_f_random) = evaluation.f1_score(y_test, random_predictions)
    (f_knn, macro_f_knn) = evaluation.f1_score(y_test, knn_predictions)

    assert np.allclose(f_random, np.array(
        [0.33333333, 0.23529412, 0.31578947]))
    assert np.isclose(macro_f_random, 0.29480564155486755)

    assert np.allclose(f_knn, np.array([1., 0.94117647, 0.95238095]))
    assert np.isclose(macro_f_knn, 0.9645191409897292)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
