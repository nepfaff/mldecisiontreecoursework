import numpy as np
from entropy import evaluate_entropy


def evaluate_information_gain(
    y: np.ndarray, split_point: float, y_entropy: float = None
) -> float:
    """
    Evaluates the Information gained from splitting labels in 'y' into left_labels
    and right_labels.

    :param y: Class labels of shape (n,). Each label is an outcome of the
        random variable Y.
    :param split_point: Index where labels are divided into left_labels and
        right_labels. The right_label contains the split index.
    :param y_entropy: The entropy of 'y'.
    :return: The computed information gain.
    """

    # Calculate the entropy in 'y' if it is not given.
    if y_entropy is None:
        y_entropy = evaluate_entropy(y)

    # Divide dataset into two parts to evaluate information gained from doing so
    left_labels = y[:split_point]
    right_labels = y[split_point:]

    # Evaluate entropy of left and right labels
    left_entropy = evaluate_entropy(left_labels)
    right_entropy = evaluate_entropy(right_labels)

    # Evaluate information gained using entropy and probability values of the sets of labels
    information_gain = (
        y_entropy
        - (len(left_labels) / len(y)) * left_entropy
        - (len(right_labels) / len(y)) * right_entropy
    )

    return information_gain
