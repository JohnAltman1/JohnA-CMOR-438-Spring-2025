# Random Forests

Random Forests are an ensemble method that builds multiple decision trees on random subsets of data and combines their predictions to improve accuracy and reduce overfitting.

## Key Features
- **Ensemble of Trees**: Combines results from many decision trees.
- **Random Subsampling**: Each tree sees a different subset of the data.

## Bagging (Bootstrap Aggregating)
Bagging is a general technique that trains models on different random samples (with replacement) and averages their predictions to reduce variance and overfitting.

- **Bootstrap Samples**: Each model is trained on a unique subset.
- **Parallel Training**: Models run independently.
- **Variance Reduction**: Averages out errors from individual models.

Random Forests make use of bagging, with decision trees as the base models.

## Dataset

This is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from processing the original images. In this section we specificaly looks at the "variance" and "skewness" datavalues. Variance is how much the pixels in the same image differ from each other, while skewness is how much the pixels in an image differ from a regular normal distribution.

## Reproducing Results

This should only require running all cells of the notebook. How quickly it converges does change some based on its initial random state, but that should not effect the model too much.
