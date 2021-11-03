from typing import Tuple

import numpy as np


def load_txt_data(path: str, attribute_number: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads data from a text file. The following data format is assumed:
    A number of attributes of type float followed by one label of type int.
    Individual columns should be separated by whitespace.

    :param path: The path to the data txt file.
    :param attribute_number: The number of attributes that the data contains.

    :return: A tuple of (x, y):
        - x: Contains the attributes. An array of shape (n, k) where n is the
            number of instances in the txt file and k is the 'attribute_number'.
        - y: Contains the indices to reconstruct the original array from the y_unique. An array of shape (n,).
    """

    data = np.loadtxt(path)

    # Split data into attributes and labels
    x = data[:, :attribute_number]

    # y_lables: a numpy array with shape (n, ), and each element is the label corresponding
    # to the nth instance in x.
    y = data[:, -1]

    return x, y
