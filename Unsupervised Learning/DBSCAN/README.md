# DBSCAN

DBSCAN, short for Density-Based Spatial Clustering of Applications with Noise, is a clustering algorithm that identifies areas of high density as clusters and separates low-density areas as noise or outliers.

## How DBSCAN Operates

1. **Density Criteria**:
   - A point qualifies as a **core point** if there are at least `min_samples` within a given distance `eps`.
   - Points within `eps` of a core point are considered **reachable**.
2. **Cluster Formation**:
   - Clusters grow from core points by connecting to reachable points.
   - Points not reachable from any core are marked as **noise**.

## Highlights

- **No Need to Predefine Cluster Count**: Unlike k-means, DBSCAN determines the number of clusters based on the data’s structure.
- **Noise-Aware**: Capable of isolating outliers that don’t belong to any cluster.
- **Flexible Shapes**: Can detect clusters of arbitrary shapes and sizes, making it useful for complex datasets.

DBSCAN is a strong choice for unsupervised learning problems where data distribution is non-uniform and noise is expected.


## Dataset

The DBSCAN algorithm is applied to the **Seeds** dataset, sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds). This dataset has 3 groups of diffferent wheat seed varieties, providing data on the shape and characteristics of each seed. Specific to this implementation, we are looking at the Area and Asymmetry of the seads. Area is the area the seed takes up in an image, while Asymmetry is a measure of how irregular the seed's shape is. A major reason I chose this dataset is because it had 3 groups and I wanted to work with more than just binary classification.

## Reproducing Results

This should only require running all cells of the notebook.