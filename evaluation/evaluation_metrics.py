import numpy as np


class Evaluation:
    """
    A class used to compute the evaluation metrics given a
    confusion_matrix (np.ndarray of shape (C, C)).
    """

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        self.accuracy = self._evaluate_accuracy()
        self.precisions = self._evaluate_precisions()
        self.recalls = self._evaluate_recalls()
        self.f1s = self._evaluate_f1s()

    def _evaluate_accuracy(self) -> float:
        """
        Compute the accuracy given a confusion matrix

        :return: the accuracy. Accuracy is calculated as the sum of
            diagonal elemnts over sum of all elemnts in the matrix.
        """

        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    def _evaluate_precisions(self) -> np.array:
        """
        Compute the precision per class given a confusion matrix

        :return: np.array containing the precision (float) per class.
        """
        precisions = np.empty(len(self.confusion_matrix))

        column_sums = np.sum(self.confusion_matrix, axis=0)

        for i in range(len(self.confusion_matrix)):
            precisions[i] = self.confusion_matrix[i, i] / column_sums[i]

        return precisions

    def _evaluate_recalls(self) -> np.array:
        """
        Compute the recall per class given a confusion matrix

        :return: np.array containing the recall (float) per class.
        """
        recalls = np.empty(len(self.confusion_matrix))

        row_sums = np.sum(self.confusion_matrix, axis=1)

        for i in range(len(self.confusion_matrix)):
            recalls[i] = self.confusion_matrix[i, i] / row_sums[i]

        return recalls

    def _evaluate_f1s(self) -> np.array:
        """
        Compute the f1 measure per class given a confusion matrix

        :return: np.array containing the f1 measure (float) per class.
        """
        f1s = (2 * self.precisions * self.recalls) / (self.precisions + self.recalls)
        return f1s
