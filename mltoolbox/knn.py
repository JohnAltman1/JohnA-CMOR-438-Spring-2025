import numpy as np


class KNN(object):
    """
    KNN (K-Nearest Neighbors) Classifier/Regressor
    This class implements the K-Nearest Neighbors algorithm.

    Attributes:
        x (list or numpy.ndarray): The training feature set.
        y (list or numpy.ndarray): The training labels corresponding to the feature set.
        distance (callable): A function to compute the distance between two points.
    Methods:
        k_nearest_neighbors(point, k):
            Finds the k nearest neighbors to a given point based on the distance metric.
        KNN_Predict(point, k, regression=False):
            Predicts the label or value for a given point using the k nearest neighbors.
            If `regression` is False, it performs classification by majority voting.
            If `regression` is True, it performs regression by averaging the neighbors' values.
        classification_error(test_features, test_labels, k):
            Computes the classification error on a test dataset.
            Returns the proportion of misclassified points.
    """
    def __init__(self, X_train, y_train, distance):
        """
        Initializes the k-Nearest Neighbors (k-NN) model with training data and a distance metric.

        Args:
            X_train (array-like): The training data features.
            y_train (array-like): The training data labels.
            distance (callable): A function to compute the distance between two data points.

        Attributes:
            x (array-like): Stores the training data features.
            y (array-like): Stores the training data labels.
            distance (callable): Stores the distance function.
        """
        self.x = X_train
        self.y = y_train
        self.distance = distance

    def k_nearest_neighbors(self, point, k):
        """
        Find the k-nearest neighbors to a given point based on the distance metric.
        Args:
            point (iterable): The target point for which the nearest neighbors are to be found.
            k (int): The number of nearest neighbors to retrieve.
        Returns:
            list: A list of the k-nearest neighbors, where each neighbor is represented 
                  as a list containing the point, its label, and the distance to the target point.
                  The list is sorted in ascending order of distance.
        """
        # Create an empty list to store neighbors and distances
        neighbors = []
        
        for p, label in zip(self.x, self.y):
            d = self.distance(point, p)
            temp_data = [p, label, d]
            neighbors.append(temp_data)
            
        neighbors.sort(key = lambda x : x[-1])
        
        return neighbors[:k]
    
    def KNN_Predict(self, point, k, regression = False):
        """
        Predicts the label or value for a given data point using the k-Nearest Neighbors algorithm.
        Args:
            point (iterable): The data point for which the prediction is to be made.
            k (int): The number of nearest neighbors to consider.
            regression (bool, optional): If True, performs regression by averaging the values of the neighbors.
                                          If False, performs classification by selecting the most common label.
                                          Defaults to False.
        Returns:
            int/float: The predicted label (for classification) or the predicted value (for regression).
        """

        neighbors = self.k_nearest_neighbors(point, k)
        
        if regression == False:
            labels = [x[1] for x in neighbors]
            return max(labels, key = labels.count)
        
        else:
            return sum(x[1] for x in neighbors)/k

        
    def classification_error(self, test_features, test_labels, k):
        """
        Calculate the classification error for a k-Nearest Neighbors (k-NN) model.

        Args:
            test_features (list or numpy.ndarray): A list or array of feature vectors 
                representing the test dataset.
            test_labels (list or numpy.ndarray): A list or array of true labels 
                corresponding to the test dataset.
            k (int): The number of nearest neighbors to consider for the k-NN prediction.

        Returns:
            float: The classification error, calculated as the proportion of 
                misclassified points in the test dataset.
        """
        error = 0
        for point, label in zip(test_features, test_labels):
            error += label != self.KNN_Predict(point, k)
        return error/len(test_features)
    

