import numpy as np

class Perceptron:
    """
    Perceptron Classifier

    A simple implementation of a perceptron classifier using a single-layer neural network.
    """

    def __init__(self, eta=0.5, epochs=50):
        """
        Initializes the Perceptron model.

        :param eta: Learning rate (default: 0.5)
        :param epochs: Number of epochs for training (default: 50)
        """
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        """
        Fits the Perceptron model to the training data.

        :param X: Feature array (2D numpy array)
        :param y: Label vector (1D numpy array)
        :return: The trained Perceptron instance
        """
        self.w_ = np.random.rand(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (self.predict(xi) - target)
                self.w_[:-1] -= update * xi
                self.w_[-1] -= update
                errors += int(update != 0)
            if errors == 0:
                return self
            else:
                self.errors_.append(errors)

        return self

    def net_input(self, X):
        """
        Calculates the net input for the given features.

        :param X: Feature array (1D numpy array)
        :return: The net input value
        """
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        """
        Predicts the label for the given features.

        :param X: Feature array (1D numpy array)
        :return: Predicted label (1 or -1)
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
