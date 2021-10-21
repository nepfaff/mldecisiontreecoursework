import numpy as np
from information import evaluate_information


def evaluate_entropy(y: np.ndarray) -> float:
    """
    Evaluates the entropy of a Random variable Y.

    :param y: Class labels of shape (n,). Each label is an outcome of the random variable Y.
    :return: a float entropy  :
        - entropy: the Shannon's entropy of the random variable Y.
    """

    (unique_y, unique_counts) = np.unique(y, return_counts=True)

    # evaluate the probability of each outcome y_i as (#y_i)/(#total outcomes)
    probabilities = unique_counts / len(y)
    print(unique_y)
    print(probabilities)

    entropy = 0
    for prob in probabilities:
        entropy += prob * evaluate_information(prob)

    return entropy


if __name__ == "__main__":
    y = np.array([1, 0, 1, 1, 2])
    evaluate_entropy(y)
