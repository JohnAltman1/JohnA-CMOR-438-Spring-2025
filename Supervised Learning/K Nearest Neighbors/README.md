# K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that predicts the class or value of a data point based on its `k` nearest neighbors in the feature space.

## How It Works
1. **Data as Vectors**: Each point is represented as a feature vector.
2. **Distance Calculation**: Compute the distance (e.g., Euclidean, Manhattan) between the query point and all other points. This implementation uses Euclidean distance but can be extended.
3. **Neighbor Selection**: Identify the `k` closest points.
4. **Prediction**:
   - **Classification**: Assign the most common class.
   - **Regression**: Use the average value of neighbors.

## Key Parameters
- **k**: Number of neighbors to consider; small `k` may overfit, large `k` may oversmooth.
- **Distance Metric**: Defines similarity. Common options:
- **Euclidean Distance**: √Σ(xᵢ − yᵢ)²  
- **Manhattan Distance**: Σ|xᵢ − yᵢ|

KNN is easy to implement and interpret, but it is slow on large datasets due to the increasing number of distance calculations.


## Dataset

The KNN algorithm is applied to the **Seeds** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds). This dataset has 3 groups of diffferent wheat seed varieties, providing data on the shape and charactersisitcs of each seed. Specific to this implementation, we are looking at the Area and Asymmetry of the seads. Area is the area the seed takes up in an image, while Asymmetry is a measure of how irregular the seed's shape is. A major reason I chose this dataset is because it had 3 groups and I wanted to work with more than just binary classification.

## Reproducing Results

This should only require running all cells of the notebook.