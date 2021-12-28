import numpy as np
from numpy.random import default_rng, Generator
from .metric import accuracy
from ..classification import KNNClassifier
from ml_utils import classification


def k_fold_split(n_splits: int, n_instances: int, rng: Generator = default_rng()) -> list:
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        rng (np.random.Generator, optional): A random generator. Defaults to np.random.default_rng().

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    return np.array_split(rng.permutation(n_instances), n_splits)


def train_test_k_fold(n_folds: int, n_instances: int, rng: Generator = default_rng()) -> list:
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator. Defaults to np.random.default_rng().

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, rng)

    folds = []
    for k in range(n_folds):

        test_indices = split_indices[k]
        train_indices = np.concatenate(
            split_indices[:k] + split_indices[k + 1:]
        )

        folds.append([train_indices, test_indices])

    return folds


def cross_validation_fixed_hyperparameter(x, y, k, n_folds, rng=default_rng()) -> np.ndarray:
    accuracies = np.empty(n_folds)
    for i, (train, test) in enumerate(train_test_k_fold(n_folds, len(x), rng)):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        classifier = KNNClassifier(k)
        classifier.fit(x_train, y_train)
        y_prediction = classifier.predict(x_test)
        accuracies[i] = accuracy(y_test, y_prediction)
    return accuracies


def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with three elements: 
            - a numpy array containing the train indices
            - a numpy array containing the val indices 
            - a numpy array containing the test indices
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # TODO: Complete this
        # take the splits from split_indices and keep the k-th split as testing
        # and another split as validation
        # and concatenate the remaining k-2 splits for training

        test_indices = split_indices[k]
        val_indices = split_indices[(k + 1) % n_folds]
        train_indices = np.array([], dtype=int)
        for i in range(n_folds):
            # concatenate to training set if not validation or test
            if i not in [k, (k + 1) % n_folds]:
                train_indices = np.append(train_indices, split_indices[i])

        folds.append([train_indices, val_indices, test_indices])

    return folds


def cross_validation_general_performance(x, y, n_folds, rng=default_rng()):
    accuracies = np.zeros(n_folds)
    for i, (train, val, test) in enumerate(train_val_test_k_fold(n_folds, len(x), rng)):
        # set up the dataset for this fold
        x_train = x[train]
        y_train = y[train]
        x_val = x[val]
        y_val = y[val]
        x_test = x[test]
        y_test = y[test]

        # Perform grid search, i.e.
        # for K (number of neighbours) from 1 to 10 (inclusive)
        #     evaluate the K-NN classifiers on x_val
        #     store the accuracy and classifier for each K
        max_accuracy, best_k = 0, 0
        best_model = None
        for k in range(1, 11):
            classifier = classification.KNNClassifier(k)
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_val)
            # Select the classifier with the highest accuracy
            if accuracy(y_val, predicted) > max_accuracy:
                max_accuracy = accuracy(y_val, predicted)
                best_k = k
                best_model = classifier
        # Evaluate this classifier on x_test (accuracy)
        y_predicted = best_model.predict(x_test)
        accuracies[i] = accuracy(y_test, y_predicted)
        print(
            f'Model with hyperparameter {best_k} has the higheset accuracy {accuracies[i]} at iteration {i + 1}')
    return accuracies


def cross_validation_nested_hypertuning(x, y, n_outer_folds, n_inner_folds, rng=default_rng()):
    accuracies = np.zeros(n_outer_folds)
    for i, (train_val, test) in enumerate(train_test_k_fold(n_outer_folds, len(x), rng)):
        # set up the dataset for this fold
        x_train_val = x[train_val]
        y_train_val = y[train_val]
        x_test = x[test]
        y_test = y[test]

        # Perform grid search, i.e.
        # for K (number of neighbours) from 1 to 10 (inclusive)
        #     evaluate the K-NN classifiers on x_val
        #     store the accuracy and classifier for each K
        max_accuracy, best_k = 0, 0
        for k in range(1, 11):
            classifier = classification.KNNClassifier(k)
            accuracy_sum = 0
            for train, val in train_test_k_fold(n_inner_folds, len(x_train_val), rng):
                x_train, y_train = x_train_val[train], y_train_val[train]
                x_val, y_val = x_train_val[val], y_train_val[val]
                classifier.fit(x_train, y_train)
                predicted = classifier.predict(x_val)
                accuracy_sum += accuracy(predicted, y_val)
            # The accuracy is the averaged performance on rotated train-val dataset.
            # Select the classifier with the highest accuracy
            if accuracy_sum / n_inner_folds > max_accuracy:
                max_accuracy = accuracy_sum / n_inner_folds
                best_k = k
        # Evaluate this classifier on x_test (accuracy)
        best_model = KNNClassifier(best_k)
        best_model.fit(x_train_val, y_train_val)
        y_predicted = best_model.predict(x_test)
        accuracies[i] = accuracy(y_test, y_predicted)
        print(
            f'Model with hyperparameter {best_k} has the higheset accuracy {accuracies[i]} at iteration {i + 1}')
    return accuracies
