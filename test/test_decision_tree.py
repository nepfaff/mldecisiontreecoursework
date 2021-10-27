from typing import Dict, Tuple, List

import pytest
import numpy as np

from decision_tree import find_split, decision_tree_learning, decision_tree_predict


@pytest.fixture
def find_split_datasets() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[int], List[float]
]:
    """
    :return: A tuple of (X, Y, first_split_attribute, first_split_value):
        - X: A list of instance attribute data sets ([x1, x2, x3, ...]) that have
            shape (n, k) where n is the number of instances and k the number of attributes.
        - Y: A list of data sets ([y1, y2, y3, ...]) containing the class labels
            corresponding to 'X' that have shape (n,).
        - first_split_attributes: A list containing the first split attributes corresponding to X, Y.
        - first_split_values: A list containing the first split values corresponding to X, Y.
    """

    X = [
        np.array(
            [
                [1, 1],
                [1, 2],
                [1, 4],
                [1, 5],
                [2, 1],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 5],
                [4, 2],
                [4, 3],
                [4, 5],
                [5, 1],
                [5, 4],
                [5, 5],
                [6, 2],
                [6, 3],
                [6, 5],
            ]
        ),
        np.array(
            [
                [1, 1],
                [2, 1],
                [4, 1],
                [5, 1],
                [1, 2],
                [3, 2],
                [4, 2],
                [1, 3],
                [2, 3],
                [4, 3],
                [5, 3],
                [2, 4],
                [3, 4],
                [5, 4],
                [1, 5],
                [4, 5],
                [5, 5],
                [2, 6],
                [3, 6],
                [5, 6],
            ]
        ),
    ]
    Y = [
        np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2]),
        np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2]),
    ]
    first_split_attributes = [0, 1]
    first_split_values = [4.5, 4.5]

    return X, Y, first_split_attributes, first_split_values


def test_find_split(find_split_datasets):
    """
    Tests 'find_split'.
    """

    X, Y, first_split_attributes, first_split_values = find_split_datasets
    for i in range(len(X)):
        attribute, value = find_split(X[i], Y[i])
        assert (
            attribute == first_split_attributes[i] and value == first_split_values[i]
        ), f"test case {i} failed"


@pytest.fixture
def decision_tree_learning_datasets() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[Dict], List[int]
]:
    """
    :return: A tuple of (X, Y, first_split_attribute, first_split_value):
        - X: A list of instance attribute data sets ([x1, x2, x3, ...]) that have
            shape (n, k) where n is the number of instances and k the number of attributes.
        - Y: A list of data sets ([y1, y2, y3, ...]) containing the class labels
            corresponding to 'X' that have shape (n,).
        - decision_trees: A list containing the decision trees corresponding to X, Y.
        - decision_tree_depths: A list containing the decision tree depths corresponding to X, Y.
    """

    X = [
        np.array(
            [
                [1, 1],
                [1, 2],
                [1, 4],
                [1, 5],
                [2, 1],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 5],
                [4, 2],
                [4, 3],
                [4, 5],
                [5, 1],
                [5, 4],
                [5, 5],
                [6, 2],
                [6, 3],
                [6, 5],
            ]
        )
    ]
    Y = [np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]
    decision_trees = [
        {
            "is_leaf": False,
            "attribute": 0,
            "value": 4.5,
            "left": {
                "is_leaf": False,
                "attribute": 0,
                "value": 1.5,
                "left": {"is_leaf": True, "label": 1},
                "right": {
                    "is_leaf": False,
                    "attribute": 1,
                    "value": 1.5,
                    "left": {"is_leaf": True, "label": 2},
                    "right": {
                        "is_leaf": False,
                        "attribute": 0,
                        "value": 2.5,
                        "left": {"is_leaf": True, "label": 1},
                        "right": {
                            "is_leaf": False,
                            "attribute": 1,
                            "value": 4.5,
                            "left": {
                                "is_leaf": False,
                                "attribute": 0,
                                "value": 3.5,
                                "left": {
                                    "is_leaf": False,
                                    "attribute": 1,
                                    "value": 3.0,
                                    "left": {"is_leaf": True, "label": 1},
                                    "right": {"is_leaf": True, "label": 2},
                                },
                                "right": {"is_leaf": True, "label": 2},
                            },
                            "right": {"is_leaf": True, "label": 1},
                        },
                    },
                },
            },
            "right": {"is_leaf": True, "label": 2},
        }
    ]
    decision_tree_depths = [8]

    return X, Y, decision_trees, decision_tree_depths


def test_decision_tree_learning(decision_tree_learning_datasets):
    """
    Tests 'decision_tree_learning'.
    """

    X, Y, decision_trees, depths = decision_tree_learning_datasets
    for i in range(len(X)):
        decision_tree, depth = decision_tree_learning(X[i], Y[i])
        assert (
            decision_tree == decision_trees[i] and depth == depths[i]
        ), f"test case {i} failed"


@pytest.fixture
def decision_tree_predict_datasets() -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]
]:
    """
    :return: A tuple of (X_train, Y_train, X_predict, Y_predict):
        - X_train: A list of instance attribute training data sets ([x1, x2, x3, ...]) that have
            shape (n, k) where n is the number of instances and k the number of attributes.
        - Y_train: A list of training data sets ([y1, y2, y3, ...]) containing the class labels
            corresponding to 'X_train' that have shape (n,).
        - X_predict: A list of instance attribute predict data sets ([x1, x2, x3, ...]) that have
            shape (n, k) where n is the number of instances and k the number of attributes.
        - Y_predict: A list of predict data sets ([y1, y2, y3, ...]) containing the class labels
            corresponding to 'X_predict' that have shape (n,).
    """

    X_train = [
        np.array(
            [
                [1, 1],
                [1, 2],
                [1, 4],
                [1, 5],
                [2, 1],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 5],
                [4, 2],
                [4, 3],
                [4, 5],
                [5, 1],
                [5, 4],
                [5, 5],
                [6, 2],
                [6, 3],
                [6, 5],
            ]
        )
    ]
    Y_train = [np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]
    X_predict = [
        np.array(
            [
                [1, 1],
                [1, 2],
                [1, 4],
                [1, 5],
                [2, 1],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 4],
                [3, 5],
                [4, 2],
                [4, 3],
                [4, 5],
                [5, 1],
                [5, 4],
                [5, 5],
                [6, 2],
                [6, 3],
                [6, 5],
            ]
        )
    ]
    Y_predict = [np.array([1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])]

    return X_train, Y_train, X_predict, Y_predict


def test_decision_tree_predict(decision_tree_predict_datasets):
    """
    Tests 'decision_tree_predict'.
    """

    X_train, Y_train, X_predict, Y_predict = decision_tree_predict_datasets
    for i in range(len(X_train)):
        decision_tree, _ = decision_tree_learning(X_train[i], Y_train[i])
        y_predict = decision_tree_predict(decision_tree, X_predict[i])
        assert np.all(y_predict == Y_predict[i]), f"test case {i} failed"
