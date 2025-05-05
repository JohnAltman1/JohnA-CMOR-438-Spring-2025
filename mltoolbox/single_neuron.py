import numpy as np

class SingleNeuron:
    """
    A class used to represent a single artificial neuron. 

    ...

    Attributes
    ----------
    activation_function : function
        The activation function applied to the preactivation linear combination.

    w_ : numpy.ndarray
        The weights and bias of the single neuron. The last entry being the bias. 
        This attribute is created when the train method is called.

    errors_: list
        A list containing the mean squared error computed after each iteration 
        of stochastic gradient descent per epoch. 

    Methods
    -------
    train(self, X, y)
        Trains the neuron using stochastic gradient descent. Updates weights 
        and bias based on the provided feature vectors X and target labels y. 
        Tracks the mean squared error for each epoch.

    predict(self, X)
        Computes the predicted output for the given feature vectors X using 
        the current weights, bias, and activation function.
    """

    def __init__(self, activation_function, alpha=0.5, epochs=50):
        """
        Initializes the SingleNeuron instance.

        Parameters
        ----------
        activation_function : function
            The activation function to be applied to the preactivation linear combination.
        alpha : float, optional
            The learning rate for gradient descent (default is 0.5).
        epochs : int, optional
            The number of epochs for training (default is 50).
        """
        self.alpha = alpha
        self.epochs = epochs
        self.activation_function = activation_function

    def train(self, X, y):
        """
        Trains the neuron using stochastic gradient descent.

        Parameters
        ----------
        X : numpy.ndarray
            The input feature matrix where each row is a feature vector.
        y : numpy.ndarray
            The target labels corresponding to each feature vector.

        Returns
        -------
        self : SingleNeuron
            The trained SingleNeuron instance.
        """
        self.w_ = np.random.rand(1 + X.shape[1])
        self.errors_ = []
        N = X.shape[0]

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                error = (self.predict(xi) - target)
                self.w_[:-1] -= self.alpha * error * xi
                self.w_[-1] -= self.alpha * error
                errors += 0.5 * (error ** 2)
            self.errors_.append(errors / N)
        return self

    def predict(self, X):
        """
        Predicts the output for the given input feature vectors.

        Parameters
        ----------
        X : numpy.ndarray
            The input feature matrix or vector.

        Returns
        -------
        numpy.ndarray or float
            The predicted output(s) after applying the activation function.
        """
        preactivation = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(preactivation)