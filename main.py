import os
import numpy as np
import ml_utils as ml
from ml_utils import evaluation

DATASET_PATH = os.getcwd() + '/data/iris.dat'
x, y, _ = ml.read_dataset(DATASET_PATH)
rng = np.random.default_rng(60012)
x_train, x_test, y_train, y_test = ml.split_dataset(x, y, 0.2, rng)

if __name__ == '__main__':
    # accuracies = evaluation.cross_validation_general_performance(x, y, 10)
    # accuracies = evaluation.cross_validation_nested_hypertuning(x, y, 10, 5)
    # print(accuracies)
    # print(accuracies.mean())
    # print(accuracies.std())
