import numpy as np
from mltoolbox.neural import *

class DenseNetwork(object):
    def __init__(self, layers = [784, 60, 60, 26]):
        self.layers = layers
        self.W, self.B = initialize_weights(layers = self.layers)

    def train(self, X_train, y_train, alpha = 0.046, epochs = 4):
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
        depth = len(self.layers)
        _, A = forward_pass(self.W, self.B, xi)
        return np.argmax(A[-1])