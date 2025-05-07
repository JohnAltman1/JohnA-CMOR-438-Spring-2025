# mltoolbox Python Package

This Python package is a collection of machine learning models developed over the course of a school semester. The goal of this project was twofold: to provide convenient, generalized implementations of several core machine learning algorithms, and to gain hands-on experience in designing Python classes and organizing a modular Python package.

### Single Neuron Models

Implemented in `single_neuron.py`, with supporting functions in:

- `activation_functions.py` – contains various activation functions
- `cost_functions.py` – contains cost/loss functions

Algorithms:
- **Perceptron**
- **Linear Regression**
- **Logistic Regression**

### Neural Network

Implemented in `network_class.py`, this module contains a simple feedforward Dense Neural Network implementation, with supporting functions located in `neural.py`

### K-Nearest Neighbors (KNN)

Implemented in `knn.py`, this module implements the K-nearest neighbors algorithm for classification tasks.

### K-Means Clustering

Implemented in `kmeans.py`, this module implements the K-means clustering algorithm for unsupervised learning.

### Singular Value Decomposition (SVD) Image Compression

Implemented in `svd_compression.py`, this module applies SVD to compress grayscale images, demonstrating dimensionality reduction techniques in a practical context.



## Usage

To use any of the models, simply import the package, all classes and functions are dunmped into the \__init__.py file. 
Example usage is:

```python
import mltoolbox as ml
model = ml.Perceptron_Neuron()
model.train(X, y)

