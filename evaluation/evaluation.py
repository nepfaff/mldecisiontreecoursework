import numpy as np
from numpy.random import default_rng
from decision_tree import (
    decision_tree_learning,
    decision_tree_predict,
    decision_tree_pruning,
)
from evaluation.evaluation_metrics import Evaluation


def nested_cross_validation(
    x: np.ndarray, y: np.ndarray, folds: int = 10, random_generator=default_rng()
) -> Evaluation:
    """
    Applies a nested cross validation to a dataset returning evaulation metrics

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :param folds: number of folds that the dataset is divided in.
    :param random_generator (np.random.Generator): A random generator

    :return: Instance of a class Evaluation. Evaluation contains:
        - Confusion matrix  (nparray of floats)
        - Accuracy (float)
        - Recall rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - Precision rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - F1 measures per class (nparray of floats). The ith entry is the f2 measure of the ith class.
    """
    # Randomise data & split code into folds (10) segments (of array)

    split_indices = j_fold_split(len(x), folds)
    class_labels = np.unique(y)

    # Construct a array of confusion matrices containing a confusion matrix for each test set
    average_pruned_confusion_matrices = np.empty(
        (folds, len(class_labels), len(class_labels))
    )

    for i, fold in enumerate(split_indices):
        # Assign test and train data
        test_indices = fold
        train_validation_indices = np.hstack(split_indices[:i] + split_indices[i + 1 :])

        pruned_confusion_matrices = np.empty(
            (len(train_validation_indices), len(class_labels), len(class_labels))
        )

        for nested_i, nested_fold in enumerate(train_validation_indices):
            validation_indices = nested_fold
            train_indices = np.hstack(
                train_validation_indices[:nested_i] + train_validation_indices[i + 1 :]
            )

            # Build tree using train dataset
            (decision_tree, _) = decision_tree_learning(
                x[train_indices], y[train_indices]
            )

            # Prune tree using validation dataset
            pruned_decision_tree, _ = decision_tree_pruning(
                decision_tree,
                x[train_indices],
                y[train_indices],
                x[validation_indices],
                y[validation_indices],
            )

            # Run decision tree on test data
            y_predicted = decision_tree_predict(pruned_decision_tree, x[test_indices])

            # Evaluate tree to obtain & store the confusion Matrix using test data
            pruned_confusion_matrices[nested_i] = construct_confusion_matrix(
                y[test_indices], y_predicted
            )

        # Obtain average confusion matrix from pruned inner loop of 9 confusion matrices
        average_pruned_confusion_matrices[i] = np.mean(pruned_confusion_matrices)

    # Obtain average confusion Matrix
    average_confusion_matrix = np.mean(average_pruned_confusion_matrices)

    # Obtain & return other evaluation metrics
    evaluated_algorithm = Evaluation(average_confusion_matrix)

    return evaluated_algorithm


def cross_validation(
    x: np.ndarray, y: np.ndarray, folds: int = 10, random_generator=default_rng()
) -> Evaluation:
    """
    Applies a cross validation to a dataset returning average evaulation metrics

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :param folds: number of folds that the dataset is divided in.
    :param random_generator (np.random.Generator): A random generator

    :return: Instance of a class Evaluation. Evaluation contains:
        - Confusion matrix  (nparray of floats)
        - Accuracy (float)
        - Recall rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - Precision rates per class (nparray of floats). The ith entry is the accuracy of the ith class.
        - F1 measures per class (nparray of floats). The ith entry is the f2 measure of the ith class.
    """
    # Randomise data & split code into j folds
    split_indices = j_fold_split(len(x), folds)

    class_labels = np.unique(y)

    # Construct a array of confusion matrices containing a confusion matrix for each test set
    confusion_matrices = np.empty((folds, len(class_labels), len(class_labels)))

    for i, fold in enumerate(split_indices):
        # Assign test and train data
        test_indices = fold
        train_indices = np.hstack(split_indices[:i] + split_indices[i + 1 :])

        # Build tree using train dataset
        (decision_tree, _) = decision_tree_learning(x[train_indices], y[train_indices])

        # Run decision tree on test data
        y_predicted = decision_tree_predict(decision_tree, x[test_indices])

        # Evaluate tree to obtain & store the confusion Matrix using test data

        confusion_matrices[i] = construct_confusion_matrix(
            y[test_indices], y_predicted, class_labels
        )

    # Obtain average confusion Matrix
    average_confusion_matrix = np.mean(confusion_matrices)

    # Obtain & return other evaluation metrics
    evaluated_algorithm = Evaluation(average_confusion_matrix)

    return evaluated_algorithm


def j_fold_split(
    n_instances: int, j: int = 10, random_generator=default_rng()
) -> np.ndarray:
    """
    Randomises indices and splits them into j folds
    :param n_instances: Number of instances of the dataset.
    :param j: Number of folds for splitting.
    :param random_generator (np.random.Generator): A random generator
    :return: an nparray of size j. Each element in the array is a numpy array giving the indices of the instance in that fold.

    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, j)

    return split_indices


def construct_confusion_matrix(y_gold, y_prediction, class_labels=None) -> np.ndarray:
    """
    Compute the confusion matrix.

    :param: y_gold : np.ndarray of shape (n,) the correct ground truth/gold standard labels
    :param: y_prediction : np.ndarray of shape (n,) the predicted labels
    :param: class_labels : np.ndarray of unique class labels.

    :return: np.ndarray : np.ndarray shape (C, C), where C is the number of classes. Rows are ground truth per class,
    columns are predictions

    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = y_gold == label
        gold = y_gold[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion
