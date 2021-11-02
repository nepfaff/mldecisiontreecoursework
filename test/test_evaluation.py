import numpy as np
from evaluation import cross_validation, nested_cross_validation, Evaluation


x = np.array(
    [
        [-64, -56, -61, -66, -71, -82, -81],
        [-68, -57, -61, -65, -71, -85, -85],
        [-63, -60, -60, -67, -76, -85, -84],
        [-61, -60, -68, -62, -77, -90, -80],
        [-63, -65, -60, -63, -77, -81, -87],
        [-64, -55, -63, -66, -76, -88, -83],
        [-65, -61, -65, -67, -69, -87, -84],
        [-61, -63, -58, -66, -74, -87, -82],
        [-65, -60, -59, -63, -76, -86, -82],
        [-62, -60, -66, -68, -80, -86, -91],
        [-67, -61, -62, -67, -77, -83, -91],
        [-65, -59, -61, -67, -72, -86, -81],
        [-63, -57, -61, -65, -73, -84, -84],
        [-66, -60, -65, -62, -70, -85, -83],
        [-61, -59, -65, -63, -74, -89, -87],
        [-67, -60, -59, -61, -71, -86, -91],
        [-63, -56, -60, -62, -70, -84, -91],
        [-60, -54, -59, -65, -73, -83, -84],
        [-60, -58, -60, -61, -73, -84, -88],
        [-62, -59, -63, -64, -70, -84, -84],
    ]
)

y_labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# Map labels to class indexes
(_, y) = np.unique(y_labels, return_inverse=True)


# Testing cross validaiton


evaluated1 = cross_validation(x, y)
print(evaluated1)

"""
print(evaluated1.confusion_matrix)
print("precisions:", evaluated1.precisions)
print("recalls:", evaluated1.recalls)
print("accuracy:", evaluated1.accuracy)
print("f1s:", evaluated1.f1s)
"""

# Testing evaluation


confusion_matrix = np.array(
    [
        [4, 5, 6],
        [2, 1, 9],
        [12, 1, 0],
    ]
)


evaluated2 = Evaluation(confusion_matrix)
"""
print(evaluated2.confusion_matrix)
print("precisions:", evaluated2.precisions)
print("recalls:", evaluated2.recalls)
print("accuracy:", evaluated2.accuracy)
print("f1s:", evaluated2.f1s)
"""


# Testing nested cross validation

nested_evaluated = nested_cross_validation(x, y)


print(nested_evaluated.confusion_matrix)
print("precisions:", nested_evaluated.precisions)
print("recalls:", nested_evaluated.recalls)
print("accuracy:", nested_evaluated.accuracy)
print("f1s:", nested_evaluated.f1s)
