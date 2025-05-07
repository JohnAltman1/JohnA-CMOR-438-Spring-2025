# Principal Component Analysis (PCA)

Principal Component Analysis is a method used to reduce the number of variables in a dataset while retaining the most significant information. It works by projecting the data onto a new coordinate system, where the axes (called principal components) capture the directions of highest variance.

## How It Works

1. Standardize the data to have zero mean and unit variance.
2. Compute the covariance matrix to understand feature relationships.
3. Extract the eigenvectors and eigenvalues of the covariance matrix.
4. Rank components by their eigenvalues and select the top `n` components.
5. Transform the original data onto this reduced set of axes.

## Advantages

- Reduces complexity without losing much data variance.
- Accelerates training for machine learning algorithms.
- Reveals hidden structure or clusters in data.

## Drawbacks

- Components are abstract combinations—not always easy to interpret.
- May lose useful low-variance information.

## Dataset

PCA is applied to the **Seeds** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds). This dataset has 3 groups of diffferent wheat seed varieties, providing data on the shape and characteristics of each seed.

## Reproducing Results

This should only require running all cells of the notebook.