from typing import Tuple

import numpy as np
from numpy.random import default_rng


def split_dataset_into_train_and_test(
    x: np.ndarray,
    y: np.ndarray,
    test_proportion: float,
    rg: np.random.Generator = default_rng(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and test sets according to the given proportions.
    
    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :param test_proportion: The desired proportion of test examples (0.0-1.0).
    :param rg: A random generator.
    :return: Returns a tuple of (x_train, y_train, x_test, y_test):
        - x_train: Training instances of shape (n - n_test, k)
        - y_train: Training labels of shape (n - n_test,)
        - x_test: Test instances of shape (n_test, k)
        - y_test: Test labels of shape (n_test,)
    """

    # Shuffled indices for shuffling the instances
    shuffled_indices = rg.permutation(len(x))

    # Number of test instances
    n_test = round(len(x) * test_proportion)

    x_train = x[shuffled_indices[n_test:]]
    y_train = y[shuffled_indices[n_test:]]
    x_test = x[shuffled_indices[:n_test]]
    y_test = y[shuffled_indices[:n_test]]

    return x_train, y_train, x_test, y_test
