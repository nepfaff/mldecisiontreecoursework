from io import StringIO
from typing import Tuple

import pytest
import numpy as np
from data_loading import load_txt_data, split_dataset_into_train_and_test


@pytest.fixture
def dataset_txt() -> Tuple[StringIO, int, int]:
    """
    :return: A tuple of (data, rows, attributes):
        - data: A StringIO that can be used as a txt file.
        - rows: The number of rows in the StringIO.
        - attributes: The number of attributes in data.
    """

    return (
        StringIO(
            "-64 -56 -61 -66 -71 -82 -81 1\n"
            + "-68 -57 -61 -65 -71 -85 -85 1\n"
            + "-63 -60 -60 -67 -76 -85 -84 1\n"
            + "-61 -60 -68 -62 -77 -90 -80 2\n"
            + "-63 -65 -60 -63 -77 -81 -87 1\n"
            + "-64 -55 -63 -66 -76 -88 -83 1\n"
            + "-65 -61 -65 -67 -69 -87 -84 4\n"
            + "-61 -63 -58 -66 -74 -87 -82 1\n"
            + "-65 -60 -59 -63 -76 -86 -82 1\n"
            + "-62 -60 -66 -68 -80 -86 -91 3\n"
            + "-67 -61 -62 -67 -77 -83 -91 1\n"
            + "-65 -59 -61 -67 -72 -86 -81 1\n"
            + "-63 -57 -61 -65 -73 -84 -84 5\n"
            + "-66 -60 -65 -62 -70 -85 -83 1\n"
            + "-61 -59 -65 -63 -74 -89 -87 2\n"
            + "-67 -60 -59 -61 -71 -86 -91 1\n"
            + "-63 -56 -60 -62 -70 -84 -91 3\n"
            + "-60 -54 -59 -65 -73 -83 -84 2\n"
            + "-60 -58 -60 -61 -73 -84 -88 1\n"
            + "-62 -59 -63 -64 -70 -84 -84 1"
        ),
        20,
        7,
    )


@pytest.fixture
def dataset_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: A tuple of (x, y):
        - x: Contains instance attributes of shape (n, k) where n is the number
            of instances and k the number of attributes.
        - y: Contains the class labels corresponding to x of shape (n,).
    """

    x = np.array(
        [
            [-64, -56, -61, -66, -71, -82, -81],
            [-68, -57, -61, -65, -71, -85, -85],
            [-63, -60, -60, -67, -76, -85, -84],
            [-61, -60, -68, -62, -77, -90, -80],
            [-63, -65, -60, -63, -77, -81, -87],
            [-64, -55, -63, -66, -76, -88, -83],
            [-65, -61, -65, -67, -69, -87, -84],
            [-61, -63, -58, -66, -74, -87, -82],
            [-65, -60, -59, -63, -76, -86, -82],
            [-62, -60, -66, -68, -80, -86, -91],
            [-67, -61, -62, -67, -77, -83, -91],
            [-65, -59, -61, -67, -72, -86, -81],
            [-63, -57, -61, -65, -73, -84, -84],
            [-66, -60, -65, -62, -70, -85, -83],
            [-61, -59, -65, -63, -74, -89, -87],
            [-67, -60, -59, -61, -71, -86, -91],
            [-63, -56, -60, -62, -70, -84, -91],
            [-60, -54, -59, -65, -73, -83, -84],
            [-60, -58, -60, -61, -73, -84, -88],
            [-62, -59, -63, -64, -70, -84, -84],
        ]
    )
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

    return x, y


def test_load_txt_data(dataset_txt):
    """
    Test 'load_txt_data'.
    """

    path, rows, attributes = dataset_txt
    x, y = load_txt_data(path, attributes)

    # Test that x has the correct shape
    x_rows, x_columns = x.shape
    assert x_rows == rows and x_columns == attributes

    # Test that y has the correct shape
    assert len(y) == rows


def test_split_dataset_into_train_and_test(dataset_arrays):
    """
    Test 'split_dataset_into_train_and_test'.
    """

    x, y = dataset_arrays
    test_proportion = 0.2

    x_train, y_train, x_test, y_test = split_dataset_into_train_and_test(
        x, y, test_proportion
    )

    # Test that instances and labels have the same length
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)

    # Test that split proportion is correct
    assert len(x_test) == test_proportion * len(x) and (
        len(x_train) == (1 - test_proportion) * len(x)
    )
