# Decision Trees

Decision Trees split data based on feature values, forming a tree structure where each path represents a decision rule.

## Key Concepts
- **Root Node**: Represents the full dataset.
- **Internal Nodes**: Decision points based on features.
- **Leaf Nodes**: Final output (class or value).
- **Splitting**: Dividing data based on feature conditions.

## Pros
- Easy to understand and visualize.
- Very versitile, works well with irregular datasets.

## Cons
- Can overfit if too deep.
- Sensitive to small data changes.


## Dataset

This is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from procession the original images. In this section we specificaly looks at the "skewness" and "entropy" datavalues. Skewness is how much the pixels in an image differ from a regular normal distribution, while entropy is the disorder in the system, so inconsistent levels of ink, or large, complex signatures may have more entropy.

## Reproducing Results

This should only require running all cells of the notebook.
