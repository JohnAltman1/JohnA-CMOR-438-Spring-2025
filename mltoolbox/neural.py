import numpy as np

__all__ = ["sigmoid", "d_sigmoid", "mse", "initialize_weights", "forward_pass", "MSE"]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

def mse(a, y):
    return .5 * sum((a[i] - y[i])**2 for i in range(10))[0]


def initialize_weights(layers = [784, 60, 60, 26]):
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
    cost = 0.0
    m = 0
    for xi, yi in zip(X, y):
        a = forward_pass(W, B, xi, predict_vector = True)
        cost += mse(a, yi)
        m+=1
    return cost/m


