import matplotlib.pyplot as plt
import os


def joinNodes(x1: float, y1: float, x2: float, y2: float, ax: plt.Axes) -> None:
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
    current_depth: int,
    depth: int,
    width: int,
) -> None:
    """
    Recursivly plots nodes on the figure, with description of the node

    :param x: x-axis location of node.
    :param y: y-axis location of node.
    :param Node: dict containing node information.
    :param ax: contains figure elements.
    :param current depth: hold value of current depth in tree.
    :param depth: max depth of tree.
    :param width: max width of tree taking into account size of nodes.
    :return: None.
    """
    # Checking to see if leaf node

    if Node["is_leaf"]:
        box_text = f'leaf: {Node["label"]}'

        # Plotting node with a box around it
        ax.text(
            x,
            y,
            box_text,
            color="red",
            bbox=dict(facecolor="white", edgecolor="red"),
        )

    else:
        # Creating text visible for each node
        box_text = f'[X{Node["attribute"]} < {Node["value"]}]'

        # Plotting node with a box around it
        ax.text(
            x,
            y,
            box_text,
            color="red",
            bbox=dict(facecolor="white", edgecolor="red"),
        )

        # New distance calculation

        distance = width / (2 ** current_depth)
        # Checking for left and right nodes and recusivly calling function
        if Node["left"] != None:
            # Creating location of node

            plotNode(
                (x - (distance)),
                (y - 5),
                Node["left"],
                ax,
                (current_depth + 1),
                depth,
                width,
            )
            joinNodes(x, y, (x - distance), (y - 5), ax)

        if Node["right"] != None:
            plotNode(
                (x + distance),
                (y - 5),
                Node["right"],
                ax,
                (current_depth + 1),
                depth,
                width,
            )
            joinNodes(x, y, (x + distance), (y - 5), ax)


def plotTree(Tree: dict, tree_name: str = "tree", depth: int = 0) -> None:
    """
    Function that takes the top node of the tree and plots the whole tree

    :param Tree: dict containing tree.
    :param tree_name: name of .png file.
    :param depth: depth of tree.
    :return: No return
    :result: .png file named tree.png, found in visualisation folder.
    """
    # Using depth of tree to work out maximum width
    # NOTE: constants obtained via trial and error
    width = (2 ** depth) * 5

    # Defining figure information
    start_location = [0, 0]
    _, ax = plt.subplots(figsize=(100, 20))

    # Calling recursive node plotting software
    plotNode(start_location[0], start_location[1], Tree, ax, 0, depth, width)

    # Assinging figure boarders and information

    y_depth = -depth * 5

    plt.ylim([5, y_depth])
    plt.gca().invert_yaxis()
    ax.axis("off")
    plt.show()
    #   Saving figure to file
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig(f"figures/{tree_name}.png")
