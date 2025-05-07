# Perceptron Algorithm 

The perceptron algorithm is used for binary classification tasks, where the goal is to separate data points into two distinct classes using a linear decision boundary.

## Algorithm Overview

The perceptron works by iteratively adjusting its weights based on the input data and the classification error. The algorithm can be summarized as follows:

1. **Initialization**: Start with random weights **w** and bias **b**, often initialized to zero.

2. **Prediction**: For an input vector **x**, compute the weighted sum:
   
   z = w · x + b

   Apply the step function to determine the predicted class:

   ŷ = 1 if z ≥ 0  
   ŷ = 0 if z < 0

3. **Update Rule**: For each misclassified data point, update the weights and bias:

   w ← w + η(y - ŷ)x  
   b ← b + η(y - ŷ)

   Where:
   - y is the true label  
   - ŷ is the predicted label  
   - η is the learning rate

4. **Repeat**: Iterate over the dataset multiple times until the algorithm converges (i.e., no misclassifications) or a maximum number of iterations is reached.

## Key Properties

- The perceptron algorithm guarantees convergence if the data is linearly separable.
- If the data is not linearly separable, the algorithm will not converge, and the weights will oscillate.


This simple yet powerful algorithm laid the foundation for more advanced neural network architectures.


## Dataset

The perceptron is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from processing the original images. In this section we specificaly looks at the "variance" and "entropy" datavalues. Variance is how much the pixels in the same image differ from each other, so a heavilly inked signature may have more variance. Entropy is the disorder in the system, so inconsistent levels of ink, or large, complex signatures may have more entropy.

## Reproducing Results

This should only require running all cells of the notebook. How quickly it converges does change some based on its initial random state, but that should not effect the model too much.


