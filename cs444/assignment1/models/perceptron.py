"""Perceptron model."""

import numpy as np
np.random.seed(111)

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.reg_const = 8

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        obs = X_train.shape[0]
        dim = X_train.shape[1]
        
        # Random initial weights
        self.w = np.random.rand(dim, self.n_class)
        indices = np.arange(obs)

        for epoch in range(self.epochs):
            prev_y = X_train @ self.w
            tmp = [np.argmax(i) for i in prev_y]
            print("Epoch", epoch, ", Accuracy",np.sum(y_train == tmp) / len(y_train) * 100)

            np.random.shuffle(indices)

            #update weights for incorrect ones
            for x in indices:
                for c in range(self.n_class):
                    if c != y_train[x]:
                        if np.dot(np.transpose(self.w)[c], X_train[x]) > np.dot(np.transpose(self.w)[y_train[x]], X_train[x]):
                            np.transpose(self.w)[y_train[x]] = np.transpose(self.w)[y_train[x]] + self.lr * X_train[x]
                            np.transpose(self.w)[c] = np.transpose(self.w)[c] - self.lr * X_train[x]
                    np.transpose(self.w)[c] += (self.lr * self.reg_const / obs) * np.transpose(self.w)[c]
            self.lr =  self.lr * 0.85

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me           
        result =  X_test @ self.w
        return [np.argmax(p) for p in result]