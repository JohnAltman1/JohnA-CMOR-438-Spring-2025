# Ensemble Methods

Ensemble methods combine predictions from multiple models to improve performance, reduce error, and increase robustness.

## Why Use Ensembles?
- **Error Reduction**: Aggregating different models helps cancel out individual errors.
- **Robustness**: Less prone to overfitting than single models.
- **Better Predictions**: Leverages diverse model strengths for improved accuracy.

## Common Techniques

1. **Bagging (Bootstrap Aggregating)**
   - Example: Random Forests  
   - Trains multiple models on random subsets of data and averages results to reduce variance.

2. **Boosting**
   - Examples: Gradient Boosting, AdaBoost  
   - Trains models sequentially, each correcting errors from the previous one to reduce bias.

3. **Voting / Averaging**
   - Combines predictions using majority vote (classification) or average (regression).


## Dataset

This is applied to the **Banknote Authentication** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from processing the original images. In this section we specificaly looks at the "skewness" and "entropy" datavalues. Skewness is how much the pixels in an image differ from a regular normal distribution, while entropy is the disorder in the system, so inconsistent levels of ink, or large, complex signatures may have more entropy.

## Reproducing Results

This should only require running all cells of the notebook.

