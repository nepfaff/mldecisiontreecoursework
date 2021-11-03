from typing import Tuple, List

import pytest
import numpy as np

from evaluation import construct_confusion_matrix
from evaluation.evaluation_metrics import Evaluation


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
        np.array([[3, 1], [2, 2]]),
        np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]]),
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


@pytest.fixture
def evaluation_metric_data() -> Tuple[
    List[np.ndarray], List[float], List[np.ndarray], List[np.ndarray]
]:
    """
    :return: A tuple of (confusion_matrices, accuracies, precisions, recalls, f1s):
        - confusion_matrices: A list of confusion matrices where columns are predicted
            and rows are actual labels.
        - accuracies: The accuracies of the confusion matrices. Each confusion matrix is
            associated with a single accuracy value.
        - precisions: The precisions of the confusion matrices. Each confusion matrix is
            asociated with a precision np.ndarray of shape (n,) where n is the number of classes.
        - recalls: The precisions of the confusion matrices. Each confusion matrix is
            asociated with a recall np.ndarray of shape (n,) where n is the number of classes.
        - f1s: The precisions of the confusion matrices. Each confusion matrix is
            asociated with a F1-measure np.ndarray of shape (n,) where n is the number of classes.
    """

    confusion_matrices = [
        np.array([[3, 1], [2, 2]]),
        np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]]),
    ]
    accuracies = [0.625, 0.500]
    precisions = [np.array([0.600, 0.667]), np.array([0.333, 1.000, 0.500, 0.500])]
    recalls = [np.array([0.750, 0.500]), np.array([0.500, 0.500, 0.500, 0.500])]
    f1s = [np.array([0.667, 0.571]), np.array([0.400, 0.667, 0.500, 0.500])]

    return confusion_matrices, accuracies, precisions, recalls, f1s


def test_evaluation_metrics(evaluation_metric_data):
    """
    Tests for 'Evaluation'.
    """

    confusion_matrices, accuracies, precisions, recalls, f1s = evaluation_metric_data
    for i in range(len(confusion_matrices)):
        evaluation_metrics = Evaluation(confusion_matrices[i])

        assert evaluation_metrics.accuracy == pytest.approx(
            accuracies[i], rel=1e-3
        ), f"test case {i} failed; wrong accuracy"

        assert np.allclose(
            evaluation_metrics.precisions, precisions[i], atol=1e-3
        ), f"test case {i} failed; wrong precision"

        assert np.allclose(
            evaluation_metrics.recalls, recalls[i], atol=1e-3
        ), f"test case {i} failed; wrong recall"

        assert np.allclose(
            evaluation_metrics.f1s, f1s[i], atol=1e-3
        ), f"test case {i} failed; wrong F1-measure"
