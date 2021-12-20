import numpy as np


def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The path to the datset file
    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
                - x is a numpy array with shape (N, K),
                    where N is the number of instances
                    K is the number of features/attributes
                - y is a numpy array with shape (N, ), and each element should be
                    an integer from 0 to C-1 where C is the number of classes
                - classes : a numpy array with shape (C, ), which contains the
                    unique class labels corresponding to the integers in y
    """
    dataset = np.loadtxt(filepath, dtype=str)
    x, labels = [], []
    for row in dataset:
        if not isinstance(row, np.ndarray):
            row = row.strip().replace(' ', ',').split(',')
        x.append([float(e) for e in row[:-1]])
        labels.append(row[-1])
    classes, y = np.unique(labels, return_inverse=True)
    return np.array(x), y, classes


def shuffle_dataset(x, y, rng=np.random.default_rng()):
    """Shuffle dataset to mix the data with different labels.

    Args:
        x (np.ndarray): Instances, numpy array with shape (N, K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        rng (np.random.Generator, optional): A random generator. Defaults to np.random.default_rng().
    Returns:
        tuple: returns a tuple of (x_shuffle, y_shuffle)
               - x_shuffle (np.ndarray): Shuffled instances shape (N, K)
               - y_shuffle (np.ndarray): Shuffled labels, shape (N, )
    """
    N, _ = x.shape
    indices = np.arange(N)
    rng.shuffle(indices)
    return x[indices], y[indices]


def split_dataset(x, y, test_proportion, rng=np.random.default_rng()):
    """ Split dataset into training and test sets, according to the given 
        test set proportion.

    Args:
        x (np.ndarray): Instances, numpy array with shape (N,K)
        y (np.ndarray): Class labels, numpy array with shape (N,)
        test_proportion (float): the desired proportion of test examples
                                 (0.0-1.0)
        rng (np.random.Generator, optional): A random generator. Defaults to np.random.default_rng().

    Returns:
        tuple: returns a tuple of (x_train, x_test, y_train, y_test)
               - x_train (np.ndarray): Training instances shape (N_train, K)
               - x_test (np.ndarray): Test instances shape (N_test, K)
               - y_train (np.ndarray): Training labels, shape (N_train, )
               - y_test (np.ndarray): Test labels, shape (N_train, )
    """
    x_shuffle, y_shuffle = shuffle_dataset(x, y, rng)

    boundary = np.floor(len(x) * (1 - test_proportion)).astype('int64')
    return x_shuffle[:boundary], x_shuffle[boundary:], y_shuffle[:boundary], y_shuffle[boundary:]
