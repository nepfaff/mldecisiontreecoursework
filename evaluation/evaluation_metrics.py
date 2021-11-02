import numpy as np
from numpy.core.numeric import NaN


class Evaluation:
    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.accuracy = self.evaluate_accuracy()
        self.precisions = self.evaluate_precisions()
        self.recalls = self.evaluate_recalls()
        self.f1s = self.evaluate_f1s()

    def evaluate_accuracy(self) -> float:
        """Compute the accuracy given a confusion matrix

        Args:
            a confusion matrix (n, n), where n is the number of labels

        Returns:
            float : the accuracy
        """

        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    def evaluate_precisions(self) -> np.array:
        precisions = np.empty(
            (
                len(
                    self.confusion_matrix,
                )
            )
        )

        column_sum = np.sum(self.confusion_matrix, axis=0)

        for (i, row) in enumerate(self.confusion_matrix):
            precisions[i] = self.confusion_matrix[i, i] / column_sum[i]

        return precisions

    def evaluate_recalls(self) -> np.array:
        recalls = np.empty(
            (
                len(
                    self.confusion_matrix,
                )
            )
        )

        row_sum = np.sum(self.confusion_matrix, axis=1)

        for (i, row) in enumerate(self.confusion_matrix):
            recalls[i] = self.confusion_matrix[i, i] / row_sum[i]

        return recalls

    def evaluate_f1s(self):
        f1s = (2 * self.precisions * self.recalls) / (self.precisions + self.recalls)
        return f1s
