import visualisation

Node1 = {
    "is_leaf": False,
    "attribute": 2,
    "value": -5.6,
    "left": None,
    "right": None,
    "label": False,
}
Node1["left"] = {
    "is_leaf": False,
    "attribute": 1,
    "value": 77,
    "left": None,
    "right": None,
    "label": False,
}

Node1["right"] = {
    "is_leaf": False,
    "attribute": 1,
    "value": 77,
    "left": {
        "is_leaf": False,
        "attribute": 1,
        "value": 77,
        "left": None,
        "right": None,
        "label": False,
    },
    "right": None,
    "label": False,
}

visualisation.plotTree(Node1)
