# Neural Networks

Neural networks build on the concept of a **single neuron**, extending it into multi-layer systems that can model complex, non-linear patterns. These layers include an input layer, one or more hidden layers, and an output layer. While a single neuron can only perform linear separation, neural networks use layers of interconnected neurons and non-linear **activation functions** to learn intricate relationships in data.

## Key Concepts

- **Weights & Layers**: Each neuron has weights for its inputs. Except for the input layer, each layer receives input from the previous layer’s output.
- **Activation Functions**: These introduce non-linearity into the network, allowing it to model complex data beyond linear boundaries.
  
  ```
  z = ∑ w_i x_i + b
  ŷ = σ(z)
  ```

- **Forward Propagation**: Input flows through the network, layer by layer, applying weights, biases, and activation functions to produce output predictions.
- **Backpropagation**: The network learns by minimizing the prediction error using gradients and the chain rule to update weights and biases.

  ```
  δ^l = (∇ₐ C) ⊙ σ'(z^l)
  δ^l = ((W^{l+1})ᵀ δ^{l+1}) ⊙ σ'(z^l)
  ```

## Training Process

1. **Initialize** weights and biases randomly.
2. **Forward propagate** to compute predictions.
3. **Calculate error** using a cost function (e.g., MSE).
4. **Backpropagate** to update weights and biases.
5. **Repeat** for many epochs.

After training, the model is evaluated on test data to ensure it generalizes well and hasn’t **overfitted** to the training data.

## Dataset

The neural network model is applied to the **EMNIST** dataset, which is an extended dataset of handwritten characters. We only look at letters for this model, classifying the images to the appropriate letters. Notably, this dataset does not distinguish between capital and lowercase letters, so the model has to assign the characters "G" and "g" to the same output.

## Reproducing Results

This should only require running all cells of the notebook. It is useful to run the final cell multiple times to see how to model performs on the dataset.