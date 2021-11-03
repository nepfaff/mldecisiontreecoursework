from typing import Tuple, List

import pytest
import numpy as np

from evaluation import construct_confusion_matrix


@pytest.fixture
def construct_confusion_matrix_data() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    """
    :return: A tuple of (y_gold, y_prediction, confusion_matrices):
        - y_gold: A list of actual class label arrays of shape (n,).
        - y_prediction: A list of predicted class label arrays of shape (n,).
        - confusion_matrices: A list of confusion matrices where columns are predicted
            and rows are actual labels. The confusion matrices correspond to 'y_gold'
            and 'y_predicted'.
    """

    y_gold = [np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 1, 1, 2, 2, 3, 3])]
    y_prediction = [
        np.array([0, 0, 1, 0, 1, 0, 1, 0]),
        np.array([0, 3, 2, 1, 2, 0, 3, 0]),
    ]
    confusion_matrices = [
        [[3, 1], [2, 2]],
        [[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
    ]

    return y_gold, y_prediction, confusion_matrices


def test_construct_confusion_matrix(construct_confusion_matrix_data):
    """
    Tests 'construct_confusion_matrix'.
    """

    y_gold, y_prediction, confusion_matrices = construct_confusion_matrix_data
    for i in range(len(y_gold)):
        assert np.all(
            construct_confusion_matrix(y_gold[i], y_prediction[i])
            == confusion_matrices[i]
        ), f"test case {i} failed"
