"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        batch_size = X_train.shape[0]
        dim = X_train.shape[1]
        batch_w = np.zeros((dim, self.n_class))
        
        for x in range(batch_size):
            for c in range(self.n_class):
                if c != y_train[x]:
                    if np.dot(np.transpose(self.w)[y_train[x]], X_train[x]) - np.dot(np.transpose(self.w)[c], X_train[x]) < 1:
                        np.transpose(batch_w)[y_train[x]] = np.transpose(batch_w)[y_train[x]] + self.lr * X_train[x]
                        np.transpose(batch_w)[c] = np.transpose(batch_w)[c] - self.lr * X_train[x]
                np.transpose(batch_w)[c] = np.transpose(batch_w)[c] - (self.lr * self.reg_const / batch_size) * np.transpose(self.w)[c]
        return batch_w/batch_size

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labe
            s
        """
        # TODO: implement me
        # Parse dimensions
        samples = X_train.shape[0]
        dim = X_train.shape[1]

        # Set random weight
        self.w = np.random.rand(dim, self.n_class)

        # Set up mini-batch stocastic gradient descent
        batch_size = 1000
        batches = samples // batch_size
        indices = np.arange(samples)

        for epoch in range(self.epochs):
            # Print intermediate accuracy
            y_pred = X_train @ self.w
            temp = [np.argmax(i) for i in y_pred]
            print("Epoch", epoch, "Accuracy",np.sum(y_train == temp) / len(y_train) * 100)

            # Gradient descent
            np.random.shuffle(indices)
            for batch in range(batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                if end >= samples:
                    end = -1
                batch_indices = indices[start: end]
                batch_w = self.calc_gradient(X_train[batch_indices], y_train[batch_indices])
                self.w += self.lr * batch_w
            
            # Decrease learning rate
            self.lr *= 0.95


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
        y_pred = X_test @ self.w
        return [np.argmax(i) for i in y_pred]