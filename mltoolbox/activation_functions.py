import numpy as np

__all__ = ["sign_activation",
            "linear_regression_activation",
            "sigmoid_activation"]
            
def sign_activation(z):
    return np.sign(z)

def linear_regression_activation(z):
    return z

def sigmoid_activation(z):
    """
    Computes the sigmoid activation function.

    The sigmoid function is defined as:
        sigmoid(z) = 1 / (1 + exp(-z))
    It maps any real-valued number to a value between 0 and 1.

    Parameters:
    z (float or numpy.ndarray): The input value(s) to the sigmoid function. 
                                Can be a scalar or a NumPy array.

    Returns:
    float or numpy.ndarray: The computed sigmoid value(s). If the input is a scalar,
                            the output will be a scalar. If the input is an array,
                            the output will be an array of the same shape.
    """
    return 1.0/(1.0 + np.exp(-z))
