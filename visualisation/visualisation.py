import matplotlib.pyplot as plt
from typing import Literal, Tuple
import os


def joinNodes(x1: float, y1: float, x2: float, y2: float, ax: plt.Axes):
    """
    Creats a line between nodes

    :param x1: x-axis location of first node.
    :param y1: y-axis location of first node.
    :param x2: x-axis location of second node.
    :param y2: y-axis location of second node.
    :param ax: contains figure elements.
    :return: No return
    """
    ax.plot([x1, x2], [y1, y2])


def plotNode(
    x: float,
    y: float,
    Node: dict,
    ax: plt.Axes,
    division_size: float,
    distance: float = 1000,
) -> Tuple[int, int, int]:
    """
    Recursivly plots nodes on the figure, with description of the node

    :param x: x-axis location of node.
    :param y: y-axis location of node.
    :param Node: dict containing node information.
    :param ax: contains figure elements.
    :param division_size: factor that controls the size of future node distances.
    :param distance: distance between current node and children nodes.
    :return: No return
    """

    # Defining variables for creating the range of the figure
    xL_left, xR_right, y_left, y_right = 0, 0, 0, 0

    # Checking to see if leaf node
    if Node["is_leaf"] == True:
        box_text = f'leaf: {Node["label"]}'

        # Plotting node with a box around it
        ax.text(
            x, y, box_text, color="red", bbox=dict(facecolor="white", edgecolor="red")
        )

    else:
        # Creating text visible for each node
        box_text = f'[X{Node["attribute"]} < {Node["value"]}]'

        # Plotting node with a box around it
        ax.text(
            x, y, box_text, color="red", bbox=dict(facecolor="white", edgecolor="red")
        )

        # Checking for left and right nodes and recusivly calling function
        if Node["left"] != None:
            xL_left, _, y_left = plotNode(
                (x - distance),
                (y - 5),
                Node["left"],
                ax,
                division_size,
                (distance / division_size),
            )
            joinNodes(x, y, (x - distance), (y - 5), ax)

        if Node["right"] != None:
            _, xR_right, y_right = plotNode(
                (x + distance),
                (y - 5),
                Node["right"],
                ax,
                division_size,
                (distance / division_size),
            )
            joinNodes(x, y, (x + distance), (y - 5), ax)

    # Building boarders for figure

    xL = x if xL_left == 0 else xL_left

    xR = x if xR_right == 0 else xR_right

    return xL, xR, min(y_left, y_right)


def plotTree(Tree: dict, tree_name: Literal = "tree", division_size: int = 1.7):
    """
    Function that takes the top node of the tree and plots the whole tree

    :param Tree: dict containing tree.
    :param tree_name: name of .png file
    :param division_size: tuning parameter to ensure tree looks correct visualy via spacing between nodes, value must be greater than 1
    :return: No return
    :result: .png file named tree.png, found in visualisation folder.
    """
    # Defining figure information
    start_location = [0, 0]
    _, ax = plt.subplots(figsize=(50, 50))

    # Calling recursive node plotting software
    x_left, x_right, y_depth = plotNode(
        start_location[0], start_location[1], Tree, ax, division_size
    )

    # Assinging figure boarders and information

    if x_left == 0:
        x_left = -50
    if x_right == 0:
        x_right = 50

    if y_depth == 0:
        y_depth = -50

    plt.xlim([x_left, x_right])
    plt.ylim([5, y_depth])
    plt.gca().invert_yaxis()
    # ax.axis("off")
    plt.show()
    #   Saving figure to file
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig(f"figures/{tree_name}.png")
