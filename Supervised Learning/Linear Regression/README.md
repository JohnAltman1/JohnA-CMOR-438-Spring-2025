# Linear Regression Implementation

A supervised learning algorithm for fitting a linear function to data. Given feature vectors x⁽ⁱ⁾ ∈ R and target values y⁽ⁱ⁾ ∈ R, regression approximates data with a linear function, while classification separates data into groups.

## Algorithm Overview

Linear regression assumes the target output is a linear function of the input:  
ŷ⁽ⁱ⁾ = w₁x⁽ⁱ⁾ + b  

Here, w₁ is the feature weight, and b is the bias.

## Cost Function: Mean Squared Error

The Mean Squared Error (MSE) is minimized during training:  
C(w₁, b) = (1 / 2N) ∑ⁿᵢ₌₁ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²  

For a single data point (N = 1):  
C(w₁, b) = (1 / 2) (w₁x⁽ⁱ⁾ + b − y⁽ⁱ⁾)²  

## Gradient Descent Optimization

Gradient descent minimizes MSE by updating w₁ and b using their partial derivatives:  

∂C / ∂w₁ = (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)x⁽ⁱ⁾  
∂C / ∂b = (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)  

The step size, alpha, controls the update speed. Stochastic Gradient Descent (SGD) optimizes memory usage by updating weights and bias individually for each data point.

## Dataset

The perceptron is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from procession the original images. In this section we specificaly looks at the "skewness" and "curtosis" datavalues. Skewness is how much the pixels in an image differ from a redual normal distribution, while Curtosis is how "tailed" the system is, how much the weight of the edges of a distribution compare to the center of the distribution.

## Reproducing Results

This should only requier running all cells of the notebook. How quickly it converges does change some based on its initial random state, but that should not effect the model too much.