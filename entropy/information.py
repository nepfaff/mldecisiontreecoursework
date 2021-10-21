import math


def evaluate_information(probability: float) -> float:
    """
    Evaluates the information of measuring a Random variable X as outcome x.

    :param probability: The probability of X=x (i.e. P(X=x)).
    :return: a float inform  :
        - information: the information of measuring a Random variable X as outcome x.
    """

    information = -math.log(probability, 2)

    return information
