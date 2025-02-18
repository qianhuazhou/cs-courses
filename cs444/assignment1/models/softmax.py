"""Softmax model with improvements for clarity and initialization."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_batch: a numpy array of shape (N, D) containing a mini-batch of data
            y_batch: a numpy array of shape (N,) containing batch labels;
                y_batch[i] = c means X_batch[i] has label c, where 0 <= c < C

        Returns:
            Gradient with respect to weights w; an array of the same shape as w
        """
        batch_size, feature_size = X_batch.shape
        C = self.n_class

        scores = X_batch @ self.w  # Shape: (batch_size, C)

        # Stable Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)  # (batch_size, C)

        y_one_hot = np.eye(C)[y_batch]  # One-hot encoding, shape (batch_size, C)

        # Compute gradient
        grad = (X_batch.T @ (probs - y_one_hot)) / batch_size  # (feature_size, C)
        grad += self.reg_const * self.w  # Add regularization gradient

        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier using mini-batch SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, feature_size = X_train.shape
        C = self.n_class

        # Initialize weights with scaled normal distribution
        self.w = 0.01 * np.random.randn(feature_size, C)

        # Mini-batch SGD training
        batch_size = 32
        for epoch in range(self.epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                grad = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * grad  # Update weights

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict labels for test data.

        Parameters:
            X_test: a numpy array of shape (N, D) containing test data

        Returns:
            Predicted labels as a 1D array of length N
        """
        scores = X_test @ self.w
        return np.argmax(scores, axis=1)