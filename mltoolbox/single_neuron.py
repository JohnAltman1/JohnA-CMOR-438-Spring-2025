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
    SingleNeuron is a class that represents a simple single-layer neuron model. 

    Attributes:
        activation_function (callable): The activation function to be applied to the neuron's output.
        cost_function (callable): The cost function used to calculate the error during training.
    Methods:
        __init__(activation_function, cost_function):
            Initializes the SingleNeuron with the specified activation and cost functions.
        train(X, y, alpha=0.005, epochs=50):
            Trains the neuron using the provided input data and target labels.
            Args:
                X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
                y (numpy.ndarray): Target labels of shape (n_samples,).
                alpha (float): Learning rate for weight updates. Default is 0.005.
                epochs (int): Number of training iterations. Default is 50.
            Returns:
                self: The trained SingleNeuron instance.
        predict(X):
            Predicts the output for the given input data using the current weights.
            Args:
                X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
            Returns:
                numpy.ndarray: Predicted output values.
        plot_cost_function():
            Plots the cost function values recorded during training.
            Returns:
                matplotlib.figure.Figure: The figure object containing the plot.
        plot_decision_boundary(X, y, xstring="x", ystring="y"):
            Plots the decision boundary of the neuron for the given input data and labels.
            Args:
                X (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
                y (numpy.ndarray): Target labels of shape (n_samples,).
                xstring (str): Label for the x-axis. Default is "x".
                ystring (str): Label for the y-axis. Default is "y".
            Returns:
                None
    """

    def __init__(self, activation_function, cost_function):
        """
        Initializes a single neuron with the specified activation and cost functions.

        Args:
            activation_function (callable): The activation function to be used by the neuron.
            cost_function (callable): The cost function to evaluate the performance of the neuron.

        """
        self.activation_function = activation_function
        self.cost_function = cost_function

    def train(self, X, y, alpha = 0.005, epochs = 50):
        """
        Trains the single neuron model using gradient descent.

        Parameters:
        X : ndarray of shape (n_samples, n_features)
            The input data where each row represents a sample and each column represents a feature.
        y : ndarray of shape (n_samples,)
            The target values corresponding to each input sample.
        alpha : float, optional, default=0.005
            The learning rate for gradient descent.
        epochs : int, optional, default=50
            The number of iterations over the training dataset.

        Returns:
        self : object
            Returns the instance of the class after training.
        """
   
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
        """
        Predict the output for the given input data using the trained model.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Input data where each row represents a sample and each column represents a feature.

        Returns:
        array-like of shape (n_samples,)
            The predicted output for each input sample after applying the activation function.
        """
        preactivation = np.dot(X, self.w_[:-1]) + self.w_[-1]
        return self.activation_function(preactivation)


    def plot_cost_function(self):
        """
        Plots the cost function over epochs during training.

        This method generates a line plot of the cost function values 
        (stored in the `errors_` attribute) against the number of epochs. 
        It provides a visual representation of how the cost changes 
        during the training process.

        Returns:
            matplotlib.figure.Figure: The matplotlib figure object containing the plot.

        Attributes:
            errors_ (list or array-like): A sequence of cost values calculated 
                                          after each epoch during training.

        Notes:
            - The x-axis represents the epochs (starting from 1).
            - The y-axis represents the cost values.
            - The plot includes labels, a legend, and a title for better readability.
        """
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
        """
        Plots the decision boundary of the neuron model using the provided data.

        This function visualizes the decision boundary of the classifier by plotting
        the regions where the model predicts different classes. It uses the 
        `plot_decision_regions` function to create the visualization.

        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features) 
                containing the input data points.
            y (numpy.ndarray): Target labels of shape (n_samples,) corresponding 
                to the input data points.
            xstring (str, optional): Label for the x-axis. Defaults to "x".
            ystring (str, optional): Label for the y-axis. Defaults to "y".

        Returns:
            None: The function displays the plot but does not return any value.

        Notes:
            - This function assumes that the `plot_decision_regions` function is 
              available and that the current class implements a classifier interface.
            - The `clf` parameter passed to `plot_decision_regions` is set to `self`, 
              which means the current instance of the class is used as the classifier.
        """
        plt.figure(figsize = (10, 8))
        plot_decision_regions(X, y, clf = self)
        plt.title("Neuron Decision Boundary", fontsize = 18)
        plt.xlabel(xstring, fontsize = 15)
        plt.ylabel(ystring, fontsize = 15)
        plt.show()


def Perceptron_Neuron():
    """
    Creates and returns a single perceptron neuron.

    This function initializes a single neuron using a sign activation function
    and mean squared error as the loss function.

    Returns:
        SingleNeuron: An instance of a single neuron with sign activation and
                      mean squared error loss.
    """
    return SingleNeuron(sign_activation, mean_squared_error)

def Linear_Regression_Neuron():
    """
    Creates a single neuron model configured for linear regression.

    This function initializes a single neuron with a linear regression 
    activation function and a mean squared error loss function.

    Returns:
        SingleNeuron: An instance of a single neuron configured for 
                      linear regression.
    """
    return SingleNeuron(linear_regression_activation, mean_squared_error)

def Logistic_Regression_Neuron():
    """
    Creates a single neuron configured for logistic regression.

    This function initializes a single neuron with a sigmoid activation 
    function and cross-entropy loss, which are commonly used for binary 
    classification tasks in logistic regression.

    Returns:
        SingleNeuron: An instance of the SingleNeuron class configured 
        with sigmoid activation and cross-entropy loss.
    """
    return SingleNeuron(sigmoid_activation, cross_entropy_loss)