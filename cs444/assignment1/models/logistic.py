"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.
        result = np.zeros_like(z, dtype=float)

        # pos
        pos_ind = z >= 0
        result[pos_ind] = 1 / (1 + np.exp(-z[pos_ind]))

        # neg
        neg_ind = z < 0
        result[neg_ind] = np.exp(z[neg_ind]) / (np.exp(z[neg_ind]) + 1)#have to use this format to prevent overflow issue

        return result



    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        np.random.seed(43)
        data_size, feature_size = X_train.shape
        self.w = np.random.randn(feature_size)

        y = np.where(y_train == 0, -1, y_train)

        for _ in range(self.epochs):
          for i in range(data_size):
            z = np.dot(self.w, X_train[i, :])
            self.w += self.lr * self.sigmoid(-y[i] * z) * y[i] * X_train[i, :]

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        scores = np.dot(X_test, self.w)
        pos_prob = self.sigmoid(scores)
        pred_labels = (pos_prob >= self.threshold).astype(int)

        return pred_labels
