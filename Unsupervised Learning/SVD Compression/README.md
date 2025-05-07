# Singular Value Decomposition (SVD)

Singular Value Decomposition is a powerful linear algebra technique used to factor a matrix into three separate matrices that reveal important structure in the data.

For a matrix `A` of size `m x n`, SVD expresses it as:

`A = U * Σ * Vᵀ`

Where:
- `U` is an `m x m` orthogonal matrix whose columns are the left singular vectors.
- `Σ` (Sigma) is an `m x n` diagonal matrix with non-negative real numbers known as singular values, sorted in descending order.
- `Vᵀ` is the transpose of an `n x n` orthogonal matrix whose columns are the right singular vectors.

## How It Works

1. **Decomposition**: The matrix `A` is factored into `U`, `Σ`, and `Vᵀ`.
2. **Singular Values**: The diagonal entries of `Σ` (the singular values) indicate the significance of each component; larger values capture more of the original data's structure.
3. **Approximation**: By keeping only the top `k` singular values and associated vectors, we can form a low-rank approximation:

`A_k ≈ U_k * Σ_k * V_kᵀ`

Where `U_k`, `Σ_k`, and `V_kᵀ` are truncated versions that retain only the first `k` components.

## Why Use SVD?

- **Compression**: Store only the most important parts of the matrix.
- **Noise Filtering**: Remove less important components to denoise data.
- **Dimensionality Reduction**: Reduce the number of variables while preserving important patterns.



## Dataset

The SVD technique is applied to two png images of flags taken from https://hampusborgos.github.io/country-flags/. I chose flags becasue they provided a good range of image types, from super simple bicolor flags to ones with detailed designs on them. With both I can properly show the strengths and weaknesses of singular value decomposition. One last note about this data is that the images are RGB, so there is effectivly 3 matricies representing each image.

## Reproducing Results

This should only require running all cells of the notebook.