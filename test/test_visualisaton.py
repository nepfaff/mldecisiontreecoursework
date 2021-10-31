from visualisation import plotTree

# Example tree
tree = {
    "is_leaf": False,
    "attribute": 0,
    "value": 4.5,
    "left": {
        "is_leaf": False,
        "attribute": 0,
        "value": 1.5,
        "left": {"is_leaf": True, "label": 1},
        "right": {
            "is_leaf": False,
            "attribute": 1,
            "value": 1.5,
            "left": {"is_leaf": True, "label": 2},
            "right": {
                "is_leaf": False,
                "attribute": 0,
                "value": 2.5,
                "left": {"is_leaf": True, "label": 1},
                "right": {
                    "is_leaf": False,
                    "attribute": 1,
                    "value": 4.5,
                    "left": {
                        "is_leaf": False,
                        "attribute": 0,
                        "value": 3.5,
                        "left": {
                            "is_leaf": False,
                            "attribute": 1,
                            "value": 3.0,
                            "left": {"is_leaf": True, "label": 1},
                            "right": {"is_leaf": True, "label": 2},
                        },
                        "right": {"is_leaf": True, "label": 2},
                    },
                    "right": {"is_leaf": True, "label": 1},
                },
            },
        },
    },
    "right": {
        "is_leaf": False,
        "attribute": 1,
        "value": 3.0,
        "left": {
            "is_leaf": False,
            "attribute": 1,
            "value": 3.0,
            "left": {"is_leaf": True, "label": 1},
            "right": {"is_leaf": True, "label": 2},
        },
        "right": {"is_leaf": True, "label": 2},
    },
}

# Calling plotting function
plotTree(tree, "tree", 1.7)
