import numpy as np
from mltoolbox.neural import *

class DenseNetwork(object):
    """
    DenseNetwork implements a fully connected neural network (dense network) 
    with customizable architecture and training functionality. It supports forward propagation, 
    backpropagation, and prediction.

    Attributes:
        layers (list): A list of integers where each integer represents the number of neurons 
            in the corresponding layer of the network. Defaults to [784, 60, 60, 26].
        W (list): List of weight matrices initialized for each layer.
        B (list): List of bias vectors initialized for each layer.
        errors_ (list): A list that stores the mean squared error (MSE) at the end of each epoch.
    Methods:
        __init__(layers=[784, 60, 60, 26]):
            Initializes the DenseNetwork with the specified architecture. 
            Initializes weights and biases using the `initialize_weights` function.
        train(X_train, y_train, alpha=0.046, epochs=4):
            Trains the network using stochastic gradient descent (SGD) for the specified 
            number of epochs. Updates weights and biases using backpropagation.
            Args:
                X_train (array-like): Training input data.
                y_train (array-like): Training target data.
                alpha (float): Learning rate for gradient descent. Defaults to 0.046.
                epochs (int): Number of training epochs. Defaults to 4.
        predict(xi):
            Predicts the class label for a single input sample using the trained network.
            Args:
                xi (array-like): Input sample to predict.
            Returns:
                int: The predicted class label (index of the neuron with the highest activation 
                in the output layer).
    """
    def __init__(self, layers = [784, 60, 60, 26]):
        """
        Initializes the network class with the specified layer configuration.

        Args:
            layers (list, optional): A list of integers representing the number of 
                neurons in each layer of the network. Defaults to [784, 60, 60, 26].

        Attributes:
            layers (list): Stores the layer configuration of the network.
            W (list): List of weight matrices initialized for each layer.
            B (list): List of bias vectors initialized for each layer.
        """
        self.layers = layers
        self.W, self.B = initialize_weights(layers = self.layers)

    def train(self, X_train, y_train, alpha = 0.046, epochs = 4):
        """
        Trains the neural network using stochastic gradient descent (SGD).
        Parameters:
            X_train (array-like): The input training data, where each row represents a training example.
            y_train (array-like): The target values corresponding to the training data.
            alpha (float, optional): The learning rate for gradient descent. Default is 0.046.
            epochs (int, optional): The number of epochs to train the model. Default is 4.
        Returns:
            None
        Side Effects:
            - Updates the weights (`self.W`) and biases (`self.B`) of the neural network.
            - Computes and stores the mean squared error (MSE) for each epoch in `self.errors_`.
            - Prints the cost (MSE) at the start and after each epoch.
        """
        # Print the initial mean squared error
        self.errors_ = [MSE(self.W, self.B, X_train, y_train)]
        print(f"Starting Cost = {self.errors_[0]}")

        # Find your sample size
        sample_size = len(X_train)

        # Find the number of non-input layers.
        L = len(self.layers) - 1

        # For each epoch perform stochastic gradient descent. 
        for k in range(epochs):
            print("k: ",k)
            # Loop over each (xi, yi) training pair of data.
            for xi, yi in zip(X_train, y_train):
                # Use the forward pass function defined before
                # and find the preactivation and postactivation values.
                Z, A = forward_pass(self.W, self.B, xi)

                # Store the errors in a dictionary for clear interpretation
                # of computation of these values.
                deltas = dict()
                
                # print("A : ",np.matrix(A[L]).shape)
                # print("yi : ",np.matrix(yi).shape)
                # print("z : ",np.matrix(Z[L]).shape)
                # print("d_s : ",ml.d_sigmoid(np.matrix(Z[L])))

                # Compute the output error 
                output_error = np.multiply((A[L] - yi),d_sigmoid(np.matrix(Z[L])))
                deltas[L] = output_error

                # Loop from L-1 to 1. Recall the right entry of the range function 
                # is non-inclusive. 
                for i in range(L-1, 0, -1):
                    # Compute the node errors at each hidden layer
                    #deltas[i] = (self.W[i+1].T @ deltas[i+1])*ml.d_sigmoid(np.matrix(Z[i]))
                    deltas[i] = np.multiply(np.matmul(np.transpose(np.matrix(self.W[i+1])),np.matrix(deltas[i+1]))   ,   d_sigmoid(np.matrix(Z[i])))

                # Loop over each hidden layer and the output layer to perform gradient 
                # descent. 
                for i in range(1, L+1):
                    
                    self.W[i] -= alpha*deltas[i] @ A[i-1].T
                    self.B[i] -= alpha*deltas[i]

            # Show the user the cost over all training examples
            self.errors_.append(MSE(self.W, self.B, X_train, y_train))   
            print(f"{k + 1}-Epoch Cost = {self.errors_[-1]}")
    

    def predict(self, xi):
        """
        Predicts the class label for a given input sample.

        Args:
            xi (numpy.ndarray): Input feature vector of shape (n_features,).

        Returns:
            int: The predicted class label, represented as the index of the 
             maximum value in the output layer's activation.
        """
        depth = len(self.layers)
        _, A = forward_pass(self.W, self.B, xi)
        return np.argmax(A[-1])