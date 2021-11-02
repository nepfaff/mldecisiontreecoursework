from typing import Tuple
import numpy as np
from numpy.lib.function_base import average
from decision_tree import (
    decision_tree_learning,
    decision_tree_predict,
    decision_tree_pruning,
)
from numpy.random import default_rng
import evaluation_metrics


def nested_cross_validation(
    x: np.ndarray, y: np.ndarray, folds: int = 10
) -> evaluation_metrics.Evaluation:
    """
    Applies a nested cross validation to a dataset returning evaulation metrics

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :return: Instance of a class Evaluation. Evaluation contains:
        - Confusion matrix  (nparray of floats)
        - Accuracy (float)
        - Recall rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - Precision rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - F1 measures per class (nparray of floats). The ith entry is the f2 measure of the ith class.



    """
    # Randomise data & split code into folds (10) segments (of array)

    split_indicies = j_fold_split(len(x), folds)

    # Construct a array of confusion matricies containing a confusion matrix for each test set
    average_pruned_confusion_matricies = np.empty((folds,))

    for i, fold in enumerate(split_indicies):
        # Assign test and train data
        test_indicies = fold
        train_validation_indicies = (
            np.delete(split_indicies, test_indicies, axis=0)
        ).flatten()

        pruned_confusion_matricies = np.empty((9,))

        for nested_i, nested_fold in enumerate(train_validation_indicies):
            validation_indicies = nested_fold
            train_indicies = (
                np.delete(train_validation_indicies, validation_indicies, axis=0)
            ).flatten()

            # Build tree using train dataset
            (decision_tree, _) = decision_tree_learning(
                x[train_indicies], y[train_indicies]
            )

            # Prune tree using validation dataset
            pruned_decision_tree, _, _ = decision_tree_pruning(
                decision_tree,
                x[train_indicies],
                y[train_indicies],
                x[validation_indicies],
                y[validation_indicies],
            )

            # Run decision tree on test data
            y_predicted = decision_tree_predict(pruned_decision_tree, x[test_indicies])

            # Evaluate tree to obtain & store the confusion Matrix using test data
            pruned_confusion_matricies[nested_i] = construct_confusion_matrix(
                y[test_indicies], y_predicted
            )

        # TODO: Obtain average confusion matrix from pruned inner loop of 9 confusion matricies

        average_pruned_confusion_matricies[i] = (
            np.sum(pruned_confusion_matricies, axis=0) / folds
        )

    # Obtain average confusion Matrix
    average_confusion_matrix = (
        np.sum(average_pruned_confusion_matricies, axis=0) / folds
    )

    # Obtain & return other evaluation metrics
    evaluated_algorithm = evaluation_metrics.Evaluation(average_confusion_matrix)

    return evaluated_algorithm


def cross_validation(
    x: np.ndarray, y: np.ndarray, folds: int = 10
) -> evaluation_metrics.Evaluation:
    """
    Applies a cross validation to a dataset returning average evaulation metrics

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :return: Instance of a class Evaluation. Evaluation contains:
        - Confusion matrix  (nparray of floats)
        - Accuracy (float)
        - Recall rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - Precision rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - F1 measures per class (nparray of floats). The ith entry is the f2 measure of the ith class.

    """
    # Randomise data & split code into j folds

    split_indicies = j_fold_split(len(x), folds)

    # Construct a array of confusion matricies containing a confusion matrix for each test set
    confusion_matricies = np.empty((folds,))

    for i, fold in enumerate(split_indicies):
        # Assign test and train data
        test_indicies = fold
        train_indicies = (np.delete(split_indicies, test_indicies, axis=0)).flatten()

        # Build tree using train dataset
        (decision_tree, _) = decision_tree_learning(
            x[train_indicies], y[train_indicies]
        )

        # Run decision tree on test data
        y_predicted = decision_tree_predict(decision_tree, x[test_indicies])

        # Evaluate tree to obtain & store the confusion Matrix using test data

        confusion_matricies[i] = construct_confusion_matrix(
            y[test_indicies], y_predicted
        )

    # Obtain average confusion Matrix
    average_confusion_matrix = np.sum(confusion_matricies, axis=0) / folds

    # Obtain & return other evaluation metrics
    evaluated_algorithm = evaluation_metrics.Evaluation(average_confusion_matrix)

    return evaluated_algorithm


def j_fold_split(
    n_instances: int, j: int = 10, random_generator=default_rng()
) -> np.ndarray:
    """
    Randomises indicies and splits them into j folds
    :param n_instances: Number of instances of the dataset.
    :param j: Number of folds for splitting.
    :return: an nparray of size j. Each element in the array is a numpy array giving the indicies of the instance in that fold.

    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, j)

    return split_indices


def construct_confusion_matrix(y_gold, y_prediction, class_labels=None):
    """Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    for gold, predicted in zip(y_gold, y_prediction):
        # TODO: Impliment mapping to remove '-1's
        confusion[gold - 1, predicted - 1] += 1

    return confusion
