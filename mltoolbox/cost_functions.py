import numpy as np

__all__ = ["mean_squared_error",
            "cross_entropy_loss"]

def mean_squared_error(y_hat, y):
    """
    Compute the Mean Squared Error (MSE) between predicted values and actual values.

    Args:
    y_hat : array-like
        Predicted values.
    y : array-like
        Actual target values.

    Returns:
    numpy.ndarray
        The element-wise mean squared error values.

    """
    return .5*(np.array(y_hat) - np.array(y))**2

def cross_entropy_loss(y_hat, y):
    """
    Computes the cross-entropy loss between predicted probabilities and true labels.

    Args:
        y_hat (float or numpy.ndarray): The predicted probabilities for the positive class.
                                         Values should be in the range (0, 1).
        y (float or numpy.ndarray): The true labels, where 1 represents the positive class
                                     and 0 represents the negative class.

    Returns:
        float or numpy.ndarray: The computed cross-entropy loss. If inputs are arrays,
                                 the result will be an array of the same shape.
    """
    return - y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat)