from typing import Dict, Tuple

import numpy as np

from entropy import evaluate_entropy, evaluate_information_gain


def find_split(x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """
    Finds the attribute and value that results in the highest information gain.

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :return: A tuple of (attribute, value):
        - attribute: The index of the attribute in 'x' that results in the highest
            information gain.
        - value: The split value corresponding to the attribute. Splits are always
            computed using '<'.
    """

    # Calculate overall entropy
    entropy = evaluate_entropy(y)

    # Iterate over the attributes and their corresponding values
    # (every column in x represents an attribute)
    max_information_gain = -float("inf")
    for attribute, values in enumerate(x.T):
        # Sort the values to facilitate splitting
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = y[sorted_indices]

        # Find the split that maximises information gain
        for i in range(len(sorted_values) - 1):
            # Only consider splits that are between two examples
            if sorted_values[i] == sorted_values[i + 1]:
                continue

            # Calculate the information gain achieved by the split
            information_gain = evaluate_information_gain(sorted_labels, i + 1, entropy)

            # Check if the information gain achieved by this split is better than the previous one
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_split_attribute = attribute
                # Find the split between two sorted values
                best_split_value = (sorted_values[i] + sorted_values[i + 1]) / 2.0

    return best_split_attribute, best_split_value


def get_node_dict(attribute: int, value: float, left: Dict, right: Dict) -> Dict:
    """
    Returns a decision tree node dictionary.

    :param attribute: The split attribute.
    :param value: The split value.
    :param left: The left node. Corresponds to < split value.
    :param right: The right node. Corresponds to >= split value.
    """

    return {
        "is_leaf": False,
        "attribute": attribute,
        "value": value,
        "left": left,
        "right": right,
    }


def get_leaf_node_dict(label: int) -> Dict:
    """
    Returns a decision tree leaf node dictionary.

    :param label: The label of the leaf node.
    """

    return {"is_leaf": True, "label": label}


def decision_tree_learning(
    x: np.ndarray, y: np.ndarray, depth: int = 1
) -> Tuple[Dict, int]:
    """
    Returns a trained decision tree for the input training data x and y.

    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y: Class labels of shape (n,). These correspond to the instances in 'x'.
    :param depth: The depth of the decision tree. The root node is considered depth one.
    :return: A tuple of (decision_tree_dict, depth):
        - decision_tree_dict: A dictionary representing the decision tree that looks as follows:
            {
                is_leaf: False,
                attribute : 1,
                value: 54,
                left: {
                    is_leaf: False,
                    attribute : 2,
                    value: -5.6,
                    left: {
                        ...
                    },
                    right: {
                        ...
                    },
                },
                right: {
                    is_leaf: True,
                    label: 1
                },
            }
            NOTE: The value of the 'attribute' key correspond to column indices in 'x'.
        - depth: The depth of the decision tree. The root node is considered depth one.
    """

    # Check if all examples have the same label
    if np.all(y == y[0]):
        # Return a leaf node
        return get_leaf_node_dict(y[0]), depth

    # Find best split
    split_attribute, split_value = find_split(x, y)

    # Split and find corresponding left and right nodes
    left_mask = x[:, split_attribute] < split_value
    left_node, left_depth = decision_tree_learning(
        x[left_mask], y[left_mask], depth + 1
    )
    right_mask = np.bitwise_not(left_mask)
    right_node, right_depth = decision_tree_learning(
        x[right_mask], y[right_mask], depth + 1
    )

    # Return node
    return (
        get_node_dict(split_attribute, split_value, left_node, right_node),
        max(left_depth, right_depth),
    )


def decision_tree_predict(decision_tree: Dict, x: np.ndarray) -> np.ndarray:
    """
    Predict the labels for instances.

    :param decision_tree: A decision tree as produced by 'decision_tree_learning' when
        trained on data of the same format as 'x'.
    :param x: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :return: The predicted labels corresponding to 'x' of shape (n,).
    """

    y = np.empty(len(x))

    # Pass each instance through the decision tree
    for i, instance in enumerate(x):
        node = decision_tree

        # Find the leaf node corresponding to 'instance'
        while not node["is_leaf"]:
            if instance[node["attribute"]] < node["value"]:
                node = node["left"]
            else:
                node = node["right"]

        y[i] = node["label"]

    return y


def prune_node(
    node: Dict,
    y_training: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Tuple[Dict, int]:
    """
    Prunes a node with two leaf children based on improving the validation error.

    :param node: A decision tree node as produced by 'decision_tree_learning' that has two
        leaf children.
    :param y_training: Class labels of shape (n,) that were used for training the decision tree.
    :param x_validation: Attributes of shape (m, k) where n is the number of instances and k
        the number of attributes.
    :param y_validation: Class labels of shape (m,). These correspond to the instances in
        'x_validation'.
    :return: A tuple of (pruned_decision_tree, validation_errors):
        - pruned_decision_tree: A pruned version of the input 'decision_tree'.
        - validation_errors: The number of validation errors produced by the pruned tree.
    """

    # Compute the number of validation errors before pruning
    y_predict = decision_tree_predict(node, x_validation)
    validation_errors_pre_pruning = np.sum(y_predict != y_validation)

    # Find the majority class label based on the training data
    labels, counts = np.unique(y_training)
    # 'labels' can be a list or an integer
    majority_label = labels[np.argmax(counts)] if isinstance(labels, list) else labels

    # Compute the number of validation errors after pruning
    validation_errors_post_pruning = np.sum(majority_label != y_validation)

    if validation_errors_post_pruning <= validation_errors_pre_pruning:
        # Pruning improved the validation error => Prune the node
        return get_leaf_node_dict(majority_label), validation_errors_post_pruning
    # Pruning did not improve the validation error => Don't prune the node
    return node, validation_errors_pre_pruning


def decision_tree_pruning(
    decision_tree: Dict,
    x_training: np.ndarray,
    y_training: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Tuple[Dict, int, int]:
    """
    Prunes a decision tree based on improving the validation error.

    :param decision_tree: A decision tree as produced by 'decision_tree_learning' when
        trained on data of the same format as 'x_validation'.
    :param x_trianing: Attributes of shape (n, k) where n is the number of instances and k
        the number of attributes.
    :param y_training: Class labels of shape (n,). These correspond to the instances in
        'x_training'.
    :param x_validation: Attributes of shape (m, l) where n is the number of instances and k
        the number of attributes.
    :param y_validation: Class labels of shape (m,). These correspond to the instances in
        'x_validation'.
    :return: A tuple of (pruned_decision_tree, validation_errors, pruned_nodes):
        - pruned_decision_tree: A pruned version of the input 'decision_tree'.
        - validation_errors: The number of validation errors produced by the pruned tree.
    """

    # Check for leaf node
    if decision_tree["is_leaf"]:
        if len(x_validation) == 0:
            # No validation errors when there are no instances to classify
            validation_errors = 0
        else:
            y_predict = decision_tree_predict(decision_tree, x_validation)
            validation_errors = np.sum(y_predict != y_validation)

        return decision_tree, validation_errors

    # Check if the tree can be pruned
    if decision_tree["left"]["is_leaf"] and decision_tree["right"]["is_leaf"]:
        return prune_node(decision_tree, y_training, x_validation, y_validation)

    # Prune the left child tree
    left_training_mask = (
        x_training[:, decision_tree["attribute"]] < decision_tree["value"]
    )
    left_validation_mask = (
        x_validation[:, decision_tree["attribute"]] < decision_tree["value"]
    )
    left_pruned_tree, left_validation_errors = decision_tree_pruning(
        decision_tree["left"],
        x_training[left_training_mask, :],
        y_training[left_training_mask],
        x_validation[left_validation_mask, :],
        y_validation[left_validation_mask],
    )
    decision_tree["left"] = left_pruned_tree

    # Prune the right child tree
    right_training_mask = np.bitwise_not(left_training_mask)
    right_validation_mask = np.bitwise_not(left_validation_mask)
    right_pruned_tree, right_validation_errors = decision_tree_pruning(
        decision_tree["right"],
        x_training[right_training_mask, :],
        y_training[right_training_mask],
        x_validation[right_validation_mask, :],
        y_validation[right_validation_mask],
    )
    decision_tree["right"] = right_pruned_tree

    # Check if the tree can be pruned further
    if decision_tree["left"]["is_leaf"] and decision_tree["right"]["is_leaf"]:
        return prune_node(decision_tree, y_training, x_validation, y_validation)

    return decision_tree, left_validation_errors + right_validation_errors
