#!/usr/bin/env python3

from data_loading import load_txt_data
from decision_tree import (
    decision_tree_learning,
)
from visualisation import plotTree
import logging


def visualise_decision_tree(
    data_txt_path: str,
    attribute_number: int,
    data_type: str,
) -> None:
    """
    Produces a decision tree and creates visual representation of the tree
    NOTE: This function might take a while to finish.

    :param data_txt_path: The file path to the data file. The following data format is assumed:
        A number of attributes of type float followed by one label of type int. Individual
        columns should be separated by whitespace.
    :param attribute_number: The number of attributes that the data contains.
    :param data_type: type of data being used, either clean or noisy
    """

    # Load data
    x, y = load_txt_data(data_txt_path, attribute_number)

    # Obtain tree without pruning
    (decision_tree, depth) = decision_tree_learning(x, y)

    # Visualising unprunned tree
    file_name_u = f"{data_type}_unprunned"
    plotTree(decision_tree, file_name_u, depth)

    logging.info(
        "The visualised tree can be found in 'mldecisiontreecoursework/figures"
    )


def main() -> None:

    visualise_decision_trees(
        "./Data/intro2ML-coursework1/wifi_db/clean_dataset.txt", 7, "clean"
    )


if __name__ == "__main__":
    main()
