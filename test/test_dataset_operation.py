import os
import numpy as np
import pytest
import ml_utils as ml

TEST_DATA_FILEPATH = [os.getcwd() + '/data/' +
                      data for data in os.listdir('./data')]


@pytest.mark.parametrize('filepath, dimension', zip(TEST_DATA_FILEPATH, [(2000, 7, 4), (150, 4, 3), (2000, 7, 4)]))
def test_read_data(filepath, dimension):
    x, y, classes = ml.read_dataset(filepath)
    N, K, catagory = dimension
    assert x.shape == (N, K)
    assert y.shape == (N, )
    assert classes.shape == (catagory, )


@pytest.mark.parametrize('filepath', TEST_DATA_FILEPATH)
def test_shuffle_data(filepath):
    x, y, _ = ml.read_dataset(filepath)
    x_shuffle, y_shuffle = ml.shuffle_dataset(x, y)
    assert x.shape == x_shuffle.shape
    assert y.shape == y_shuffle.shape
    assert sorted(x.flatten()) == sorted(x_shuffle.flatten())
    assert sorted(y) == sorted(y_shuffle)


@pytest.mark.parametrize('filepath', TEST_DATA_FILEPATH)
def test_split_data(filepath):
    x, y, _ = ml.read_dataset(filepath)
    N, K = x.shape
    x_train, x_test, y_train, y_test = ml.split_dataset(
        x, y, test_proportion=0.2)
    assert x_train.shape == (np.floor(N * 0.8), K)
    assert x_test.shape == (np.floor(N * 0.2), K)
    assert y_train.shape == (np.floor(N * 0.8),)
    assert y_test.shape == (np.floor(N * 0.2),)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
