from typing import Tuple, List

import pytest
import numpy as np

from entropy import evaluate_entropy, evaluate_information_gain


@pytest.fixture
def evaluate_entropy_data() -> Tuple[List[np.ndarray], List[float]]:
    """
    :return: A tuple of (class_labels, entropy):
        - class_labels: A list of class label arrays of shape (n,). Each label is
            an outcome of the random variable Y.
        - entropies: A list of Shannon entropies of the random variable Y corresponding
            to 'class_labels'. The entropy values are accurate to four decimal places.
    """

    class_labels = [
        np.concatenate((np.ones(11), 2 * np.ones(9))),
        np.ones(1000),
        np.concatenate((np.zeros(500), np.ones(500))),
        np.concatenate(
            (np.zeros(500), np.ones(500), 2 * np.ones(500), 3 * np.ones(500))
        ),
        np.concatenate((np.ones(11), 2 * np.ones(9), -3 * np.ones(5))),
    ]
    entropies = [0.9928, 0.0000, 1.0000, 2.0000, 1.5161]

    return class_labels, entropies


def test_evaluate_entropy(evaluate_entropy_data):
    """
    Tests 'evaluate_entropy'.
    """

    class_labels, entropies = evaluate_entropy_data
    for i in range(len(class_labels)):
        assert (
            pytest.approx(evaluate_entropy(class_labels[i]), rel=1e-3) == entropies[i]
        ), f"test case {i} failed"


@pytest.fixture
def evaluate_information_gain_data() -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    :return: A tuple of (class_labels, split_points, information_gains):
        - class_labels: A list of class label arrays of shape (n,). Each label is
            an outcome of the random variable Y.
        - split_points: A list of indices to split the class labels at for evaluating
            information gain. Class labels are split into class_labels[:split_point]
            and class_labels[split_point:].
        - information_gains: A list of information gains corresponding to the 'class_labels'
            and the 'split_points'. Values are accurate to four decimal places.
    """

    class_labels = [
        np.concatenate((np.ones(11), 2 * np.ones(9))),
        np.ones(1000),
        np.concatenate((np.zeros(500), np.ones(500))),
        np.concatenate(
            (np.zeros(500), np.ones(500), 2 * np.ones(500), 3 * np.ones(500))
        ),
        np.concatenate((np.ones(11), 2 * np.ones(9), -3 * np.ones(5))),
    ]
    split_points = [0, 27, 89, 500, 19]
    information_gains = [0.0000, 0.0000, 0.0953, 0.8113, 0.6138]

    return class_labels, split_points, information_gains


def test_evaluate_information_gain(evaluate_information_gain_data):
    """
    Tests 'evaluate_information_gain'.
    """

    class_labels, split_points, information_gains = evaluate_information_gain_data
    for i in range(len(class_labels)):
        assert (
            pytest.approx(
                evaluate_information_gain(class_labels[i], split_points[i]), rel=1e-3
            )
            == information_gains[i]
        ), f"test case {i} failed"

