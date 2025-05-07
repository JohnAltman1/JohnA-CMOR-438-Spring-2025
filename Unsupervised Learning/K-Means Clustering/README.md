# K-Means Clustering

K-Means is a popular unsupervised learning algorithm used to organize unlabeled data into `k` distinct groups, or clusters. Each group is defined by a central point (centroid), and the goal is to minimize the distance between data points and their assigned cluster's centroid.

## Algorithm Steps

1. **Centroid Initialization**: Choose `k` initial points to act as the starting centroids.
2. **Cluster Assignment**: Assign each data point to the nearest centroid.
3. **Centroid Recalculation**: Compute new centroids by averaging the positions of all points assigned to each cluster.
4. **Iteration**: Repeat the assignment and update steps until centroids no longer change significantly or a set number of iterations is reached.

## Things to Consider

- **Choosing `k`**: The number of clusters should be defined in advance. I experiemented with a method to determine the optimal number of groups but it is still inconsistent.
- **Sensitivity to Initialization**: Results can vary depending on the initial placement of centroids.
- **Assumptions**: Works best when clusters are compact and similarly sized. Irregular cluster shapes or varying densities can reduce accuracy.
- **Outliers**: Anomalous points can skew the centroids and distort clustering.


## Dataset

The Kmeans algorithm is applied to the **Seeds** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds). This dataset has 3 groups of diffferent wheat seed varieties, providing data on the shape and charactersisitcs of each seed. Specific to this implementation, we are looking at the Area and Asymmetry of the seads. Area is the area the seed takes up in an image, while Asymmetry is a measure of how irregular the seed's shape is. A major reason I chose this dataset is because it had 3 groups and I wanted to work with more than just binary classification.

## Reproducing Results

This should only require running all cells of the notebook, but the last cell can be somewhat inconsistent so it is worthwile running it multiple times to see how it changes the result of what is the best number of clusters.