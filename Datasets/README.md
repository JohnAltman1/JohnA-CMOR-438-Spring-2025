# Datasets

### Banknote Forgeries

The data_banknote_authentication.csv dataset is sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compared forged and real banknotes with parameters gained from processing the original images.

__Used In:__
- The Perceptron
- Linear Regression
- Logistic Regression
- Random Forrests
- Ensemble
- Decision Trees


### Handwritten Letter Images

The emnist-letter.mat dataset is an extended dataset of images of handwritten characters sourced from https://www.nist.gov/itl/products-and-services/emnist-dataset. Notably, this dataset does not distinguish between capital and lowercase letters, so the model has to assign both versions of the character to the same output.

__Used In:__
- Neural Networks


### Seed Classification

The seeds_dataset.csv dataset is sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds). This dataset has 3 groups of diffferent wheat seed varieties, providing data on the shape and characteristics of each seed.

__Used In:__
- K Nearest Neighbors
- K-Means Clustering
- DBSCAN
- Principle Component Analysis


### Flag PNG Images

The flag png images in the images/ folder are sourced from https://hampusborgos.github.io/country-flags/. I chose flags for my images because they provided a good range of image types, from super simple bicolor flags to ones with detailed designs on them. With both I can properly show the strengths and weaknesses of singular value decomposition. One last note about this data is that the images are RGB, so there is effectivly 3 matricies representing each image.

__Used In:__
- Singular Value Decomposition