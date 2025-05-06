import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

from mltoolbox.activation_functions import sign_activation, linear_regression_activation, sigmoid_activation
from mltoolbox.cost_functions import mean_squared_error, cross_entropy_loss

__all__ = ["SingleNeuron", 
            "Perceptron_Neuron",
            "Linear_Regression_Neuron",
            "Logistic_Regression_Neuron"]

class SingleNeuron(object):
    """
    A class used to represent a single artificial neuron. 

    ...

    Attributes
    ----------
    activation_function : function
        The activation function applied to the preactivation linear combination.
    
    cost_function : function
        The cost function used to measure model performance.

    w_ : numpy.ndarray
        The weights and bias of the single neuron. The last entry being the bias. 
        This attribute is created when the train method is called.

    errors_: list
        A list containing the mean sqaured error computed after each iteration 
        of stochastic gradient descent per epoch. 

    Methods
    -------
    train(self, X, y, alpha = 0.005, epochs = 50)
        Iterates the stochastic gradient descent algorithm through each sample 
        a total of epochs number of times with learning rate alpha. The data 
        used consists of feature vectors X and associated labels y. 

    predict(self, X)
        Uses the weights and bias, the feature vectors in X, and the 
        activation_function to make a y_hat prediction on each feature vector. 
    """
    def __init__(self, activation_function, cost_function):
        self.activation_function = activation_function
        self.cost_function = cost_function

    def train(self, X, y, alpha = 0.005, epochs = 50):
   
        self.w_ = np.random.rand(1 + X.shape[1])
        self.errors_ = []
        N = X.shape[0]

        for _ in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                self.w_[:-1] -= alpha*(self.predict(xi) - target)*xi
                self.w_[-1] -= alpha*(self.predict(xi) - target)
                #errors += .5*((self.predict(xi) - target)**2)
                errors += self.cost_function(self.predict(xi), target)
            self.errors_.append(errors/N)
        return self

    def predict(self, X):
        preactivation = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(preactivation)


    def plot_cost_function(self):
        fig, axs = plt.subplots(figsize = (10, 8))
        axs.plot(range(1, len(self.errors_) + 1), 
                self.errors_,
                label = "Cost function")
        axs.set_xlabel("epochs", fontsize = 15)
        axs.set_ylabel("Cost", fontsize = 15)
        axs.legend(fontsize = 15)
        axs.set_title("Cost Calculated after Epoch During Training", fontsize = 18)
        return fig

    def plot_decision_boundary(self, X, y, xstring="x", ystring="y"):
        plt.figure(figsize = (10, 8))
        plot_decision_regions(X, y, clf = self)
        plt.title("Neuron Decision Boundary", fontsize = 18)
        plt.xlabel(xstring, fontsize = 15)
        plt.ylabel(ystring, fontsize = 15)
        plt.show()


def Perceptron_Neuron():
    return SingleNeuron(sign_activation, mean_squared_error)

def Linear_Regression_Neuron():
    return SingleNeuron(linear_regression_activation, mean_squared_error)

def Logistic_Regression_Neuron():
    return SingleNeuron(sigmoid_activation, cross_entropy_loss)


class SingleNeuron_:
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