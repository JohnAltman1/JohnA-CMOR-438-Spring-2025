# Logistic Regression 

This document provides an overview of the **logistic regression neuron**, including its architecture, activation function, and the loss function used during training.

## Model Overview

Logistic regression is a **binary classifier** that models the probability of a sample belonging to class `1` (positive class). It uses a **sigmoid activation function** to squash the linear combination of inputs into a range between 0 and 1.

### Prediction Function

The logistic regression neuron computes:

$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

Where:

$$
- \( \mathbf{x} \in \mathbb{R}^d \): Input features (column vector)
- \( \mathbf{w} \in \mathbb{R}^d \): Weight vector
- \( b \in \mathbb{R} \): Bias term
$$

The output is passed through the **sigmoid activation function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This produces the predicted probability $$\( \hat{y} \in (0, 1) \)$$, interpreted as:

$$
\hat{y} = P(y=1 \mid \mathbf{x}; \mathbf{w}, b)
$$

## Sigmoid Activation

The **sigmoid function** maps any real-valued input smoothly into the (0, 1) interval:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

## Dataset

The perceptron is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from procession the original images. In this section we specificaly looks at the "variance" and "entropy" datavalues. Variance is how much the pixels in the same image differ from each other, so a heavilly inked signature may have more variance. Entropy is the disorder in the system, so inconsistent levels of ink, or large, complex signatures may have more entropy.

## Reproducing Results

This should only requier running all cells of the notebook. How quickly it converges does change some based on its initial random state, but that should not effect the model too much.