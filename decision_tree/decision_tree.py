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

    # Check for identical attribute values, different label edge case
    # It is not possible to continue the tree creation
    if np.all(x == x[0]):
        # Find the majority class label
        labels, counts = np.unique(y, return_counts=True)
        # 'labels' can be a np.ndarray or an integer
        majority_label = (
            labels[np.argmax(counts)] if isinstance(labels, np.ndarray) else labels
        )

        # Return a leaf node with the value of the majority label
        return get_leaf_node_dict(majority_label), depth

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
