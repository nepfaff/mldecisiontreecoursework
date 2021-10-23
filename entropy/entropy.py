import numpy as np
import math


def evaluate_information(probability: float) -> float:
    """
    Evaluates the information of measuring a Random variable X as outcome x.

    :param probability: The probability of X=x (i.e. P(X=x)).
    :return: number of bits of information due to measuring a Random variable X as outcome x  :
        - information: the information of measuring a Random variable X as outcome x (bits).
    """

    information = -math.log(probability, 2)

    return information


def evaluate_entropy(y: np.ndarray) -> float:
    """
    Evaluates the entropy of a Random variable Y.

    :param y: Class labels of shape (n,). Each label is an outcome of the random variable Y.
    :return: The Shannon's entropy of the random variable Y  :
        - entropy: the Shannon's entropy of the random variable Y.
    """

    (_, unique_counts) = np.unique(y, return_counts=True)

    # evaluate the probability of each outcome y_i as (#y_i)/(#total outcomes)
    probabilities = unique_counts / len(y)

    entropy = np.sum(
        probabilities * np.array(list(map(evaluate_information, probabilities)))
    )

    return entropy
