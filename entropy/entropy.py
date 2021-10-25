import numpy as np


def evaluate_entropy(y: np.ndarray) -> float:
    """
    Evaluates the entropy of a Random variable Y.

    :param y: Class labels of shape (n,). Each label is an outcome of the random variable Y.
    :return: The Shannon's entropy of the random variable Y  :
        - entropy: the Shannon's entropy of the random variable Y.
    """

    (labels, labels_counts) = np.unique(y, return_counts=True)

    # evaluate the probability of labels[i]  as (labels_counts[i])/(#total outcomes)
    labels_probabilities = labels_counts / len(y)

    # (-np.log2(probabilities)) evaluates the information (in bits) of each label.
    entropy = np.sum(labels_probabilities * (-np.log2(labels_probabilities)))

    return entropy
