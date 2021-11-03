#!/usr/bin/env python3

from argparse import ArgumentParser
from statistics import mean

import numpy as np
from numpy.random import default_rng

from data_loading import load_txt_data, split_dataset_into_train_and_test
from decision_tree import (
    decision_tree_learning,
    decision_tree_pruning,
    calculate_decision_tree_depth,
)
from evaluation import cross_validation, nested_cross_validation


def evaluate_decision_tree_algorithm(
    data_txt_path: str, attribute_number: int, evaluation_results_file_path: str = None
) -> None:
    """
    Evaluates the decision tree algorithm both without pruning and with pruning.
    NOTE: This function might take a while to finish.

    :param data_txt_path: The file path to the data file. The following data format is assumed:
        A number of attributes of type float followed by one label of type int. Individual
        columns should be separated by whitespace.
    :param attribute_number: The number of attributes that the data contains.
    :param evaluation_results_file_path: The file to append the evaluation results to. The
        results are printed to the terminal if this is None.
    """

    # Load data
    x, y = load_txt_data(data_txt_path, attribute_number)

    # Evaluation without pruning
    evaluation_without_pruning = cross_validation(x, y)

    # Evaluation with pruning
    evaluation_with_pruning = nested_cross_validation(x, y)

    # Depth analysis when using 80% of the data set for training and 20% for
    # the validation set used for pruning
    tree_depths = []
    pruned_tree_depths = []
    for i in range(10):
        rg = default_rng(i)

        x_train, y_train, x_validation, y_validation = split_dataset_into_train_and_test(
            x, y, 0.2, rg
        )

        decision_tree, depth = decision_tree_learning(x_train, y_train)

        pruned_decision_tree, _ = decision_tree_pruning(
            decision_tree, x_train, y_train, x_validation, y_validation
        )
        pruned_decision_tree_depth = calculate_decision_tree_depth(pruned_decision_tree)

        tree_depths.append(depth)
        pruned_tree_depths.append(pruned_decision_tree_depth)

    # Create result text
    result_text_lines = [
        "----------------------------------------------------",
        f"Data file: {data_txt_path}",
        "Evaluation Without Pruning:\n",
        f"Confusion matrix:\n{evaluation_without_pruning.confusion_matrix}",
        f"Accuracy: {evaluation_without_pruning.accuracy}",
        *[
            f"Recall for class {i}: {evaluation_without_pruning.recalls[i]}"
            for i in range(len(evaluation_without_pruning.recalls))
        ],
        f"Macro-averaged Recall: {np.mean(evaluation_without_pruning.recalls)}",
        *[
            f"Precision for class {i}: {evaluation_without_pruning.precisions[i]}"
            for i in range(len(evaluation_without_pruning.precisions))
        ],
        f"Macro-averaged Precision: {np.mean(evaluation_without_pruning.precisions)}",
        *[
            f"F1-measure for class {i}: {evaluation_without_pruning.f1s[i]}"
            for i in range(len(evaluation_without_pruning.f1s))
        ],
        f"Macro-averaged F1-measure: {np.mean(evaluation_without_pruning.f1s)}",
        f"Average depth over 10 iterations when trained on 80% of the data set: {mean(tree_depths)}",
        "----------------------------------------------------\n",
        "Evaluation With Pruning:\n",
        f"Confusion matrix:\n{evaluation_with_pruning.confusion_matrix}",
        f"Accuracy: {evaluation_with_pruning.accuracy}",
        *[
            f"Recall for class {i}: {evaluation_with_pruning.recalls[i]}"
            for i in range(len(evaluation_with_pruning.recalls))
        ],
        f"Macro-averaged Recall: {np.mean(evaluation_with_pruning.recalls)}",
        *[
            f"Precision for class {i}: {evaluation_with_pruning.precisions[i]}"
            for i in range(len(evaluation_with_pruning.precisions))
        ],
        f"Macro-averaged Precision: {np.mean(evaluation_with_pruning.precisions)}",
        *[
            f"F1-measure for class {i}: {evaluation_with_pruning.f1s[i]}"
            for i in range(len(evaluation_with_pruning.f1s))
        ],
        f"Macro-averaged F1-measure: {np.mean(evaluation_with_pruning.f1s)}",
        f"Average depth over 10 iterations when trained on 80% of the data set: {mean(pruned_tree_depths)}",
        "----------------------------------------------------",
    ]

    # Output result text
    if evaluation_results_file_path is None:  # Print results to terminal
        for line in result_text_lines:
            print(line)
    else:  # Write results to specified file
        with open(evaluation_results_file_path, "a") as file:
            for line in result_text_lines:
                file.write(line + "\n")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="The file path to store the evaluation results in. This should be a txt file. "
        + "The results are writen to the terminal if this argument is omitted.",
        default=None,
    )
    args = parser.parse_args()

    evaluate_decision_tree_algorithm(
        "./Data/intro2ML-coursework1/wifi_db/clean_dataset.txt", 7, args.path
    )
    evaluate_decision_tree_algorithm(
        "./Data/intro2ML-coursework1/wifi_db/noisy_dataset.txt", 7, args.path
    )


if __name__ == "__main__":
    main()
