import numpy as np

__all__ = ["sigmoid", "d_sigmoid", "mse", "initialize_weights", "forward_pass", "MSE"]

def sigmoid(x):
    """
    Computes the sigmoid activation function.

    The sigmoid function is defined as:
        sigmoid(x) = 1 / (1 + exp(-x))

    It maps input values to a range between 0 and 1

    Parameters:
    x (float or numpy.ndarray): The input value or array of values.

    Returns:
    float or numpy.ndarray: The sigmoid of the input, with the same shape as `x`.
    """
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    """
    Computes the derivative of the sigmoid function.

    The derivative of the sigmoid function is given by:
    sigmoid(x) * (1 - sigmoid(x))

    Parameters:
    x (array-like): Input value or array of values for which the derivative 
                    of the sigmoid function is to be computed.

    Returns:
    array-like: The derivative of the sigmoid function applied element-wise 
                to the input.
    """
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

def mse(a, y):
    """
    Compute the Mean Squared Error (MSE) between two sequences.

    Parameters:
    a (iterable): The predicted values, expected to be an iterable of length at least 10.
    y (iterable): The true values, expected to be an iterable of length at least 10.

    Returns:
    float: The computed MSE value.
    """
    return .5 * sum((a[i] - y[i])**2 for i in range(10))[0]


def initialize_weights(layers = [784, 60, 60, 26]):
    """
    Initializes the weights and biases for a neural network with the given layer structure.
    Parameters:
        layers (list of int, optional): A list where each element represents the number of 
            neurons in a layer of the neural network. The default is [784, 60, 60, 26], 
            which corresponds to a network with an input layer of size 784, two hidden layers 
            of size 60 each, and an output layer of size 26.
    Returns:
        tuple: A tuple (W, B) where:
            - W (list of numpy.ndarray): A list of weight matrices. Each matrix connects 
                two consecutive layers in the network. The dimensions of the i-th matrix are 
                (layers[i], layers[i-1]).
            - B (list of numpy.ndarray): A list of bias vectors. Each vector corresponds to 
                a layer in the network (excluding the input layer). The dimensions of the i-th 
                vector are (layers[i], 1).
    """
    # The following Python lists will contain numpy matrices
    # connected the layers in the neural network 
    W = [[0.0]]
    B = [[0.0]]
    for i in range(1, len(layers)):
        # The scalling factor is something I found in a research paper :)
        w_temp = np.random.randn(layers[i], layers[i-1])*np.sqrt(2/layers[i-1])
        b_temp = np.random.randn(layers[i], 1)*np.sqrt(2/layers[i-1])
    
        W.append(w_temp)
        B.append(b_temp)
    return W, B

def forward_pass(W, B, xi, predict_vector = False):
    """
    Perform a forward pass through a neural network.

    Parameters:
    W : list of numpy.ndarray
        A list of weight matrices for each layer of the neural network. 
        W[i] represents the weight matrix for layer i.
    B : list of numpy.ndarray
        A list of bias vectors for each layer of the neural network. 
        B[i] represents the bias vector for layer i.
    xi : list or numpy.ndarray
        The input feature vector to the neural network.
    predict_vector : bool, optional
        If True, the function returns only the final activation (output of the last layer).
        If False, the function returns both the pre-activation values (Z) and activations (A) 
        for all layers. Default is False.

    Returns:
    tuple or numpy.ndarray
        If `predict_vector` is False, returns a tuple (Z, A):
            Z : list of numpy.ndarray
                Pre-activation values for each layer of the neural network.
            A : list of numpy.ndarray
                Activation values for each layer of the neural network.
        If `predict_vector` is True, returns only the final activation (A[-1]).
    """
    Z = [[0.0]]
    A = [xi]
    A[0] = np.transpose(np.matrix(A[0]))
    L = len(W) - 1

    for i in range(1, L + 1):
        w_ = np.matrix(W[i])
        a_ = np.matrix(A[i-1])
        b_ = np.matrix(B[i])
        z = np.add(np.matmul(w_,a_),b_)
        
        Z.append(z)
        a = sigmoid(z)
        A.append(a)


    if predict_vector == False:
        return Z, A
    else:
        return A[-1]

    

def MSE(W, B, X, y):
    """
    Computes the Mean Squared Error (MSE) for a given set of weights, biases, 
    input data, and target values.

    Parameters:
    W : array-like
        The weights of the neural network.
    B : array-like
        The biases of the neural network.
    X : iterable
        The input data, where each element corresponds to a single input sample.
    y : iterable
        The target values, where each element corresponds to the expected output 
        for the respective input sample in X.

    Returns:
    float
        The mean squared error computed over all input samples.
    """
    cost = 0.0
    m = 0
    for xi, yi in zip(X, y):
        a = forward_pass(W, B, xi, predict_vector = True)
        cost += mse(a, yi)
        m+=1
    return cost/m


