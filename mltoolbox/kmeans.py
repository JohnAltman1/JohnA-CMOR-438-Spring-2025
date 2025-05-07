import numpy as np
import matplotlib.pyplot as plt


class kmeans(object):
    """
    K-Means Clustering Implementation
    This class implements the K-Means clustering algorithm, which partitions data into k clusters 
    by iteratively assigning data points to the nearest cluster centroid and updating the centroids 
    based on the mean of the assigned points.
    Attributes:
        X (array-like): The input data points to be clustered.
        y (array-like): The labels or ground truth for the data points (if available).
        centroids (list): The current centroids of the clusters.
        distance_f (function): A function that calculates the distance between a data point and the centroids.
        distances (list): A list of distances between each data point and the centroids.
        y_hat (list): The predicted cluster labels for each data point.
    Methods:
        __init__(x, y, distance):
            Initializes the K-Means object with data, labels, and a distance function.
        make_centroids(k):
            Randomly initializes k centroids from the input data.
        assign_label():
            Assigns each data point to the nearest centroid based on the distance function.
        plot_spread(xs="Feature 1", ys="Feature 2"):
            Plots the data points and centroids in a 2D scatter plot.
        update_centroids():
            Updates the centroids by calculating the mean of the data points assigned to each cluster.
        update_loop(numLoops, solve=False, piter=True, maxIter=2000):
            Iteratively updates centroids and assigns labels for a specified number of loops.
            If `solve` is True, the loop continues until the centroids converge or the maximum number of iterations is reached.
    """
    def __init__(self,x,y,distance):
        """
        Initializes the KMeans class with the given data, labels, and distance function.

        Args:
            x (array-like): The input data points.
            y (array-like): The labels or target values corresponding to the input data.
            distance (callable): A function that computes the distance between a single data point 
                                 and each row of the centroids matrix. The function should take 
                                 two arguments: a data point and the centroids matrix.

        Attributes:
            X (array-like): Stores the input data points.
            y (array-like): Stores the labels or target values.
            centroids (list): A list to store the centroids of the clusters.
            distance_f (callable): The distance function provided during initialization.
            distances (list): A list to store the distances between data points and centroids.
            y_hat (list): A list to store the predicted cluster labels for the data points.
        """
        # distance must be a function that takes in a value x and the self.centroids matrix and give the distnaces between x and each row of the matrix 
        self.X = x
        self.y = y
        self.centroids = []
        self.distance_f = distance
        self.distances = []
        self.y_hat = []

    def make_centroids(self,k):
        """
        Initializes centroids for the k-means clustering algorithm.

        This method randomly selects `k` unique data points from the dataset `self.X`
        to serve as the initial centroids for the clustering process. The indices of
        the selected centroids are stored in a temporary list to ensure no duplicates
        are chosen.

        Args:
            k (int): The number of centroids to initialize.

        Attributes:
            self.distances (list): An empty list initialized for storing distances 
                between data points and centroids during clustering.
            self.y_hat (list): An empty list initialized for storing predicted cluster 
                labels for each data point.
            self.centroids (list): A list of `k` data points from `self.X` that are 
                randomly selected to serve as the initial centroids.

        """
        self.distances = []
        self.y_hat = []
        nl = []
        self.centroids = []
        for _ in range(k):
            i = np.random.randint(0, 100)
            while i in nl:
                i = np.random.randint(0, 100)
            nl.append(i)
            self.centroids.append(self.X[i])

    def assign_label(self):
        """
        Assigns a label to each data point in the dataset based on the nearest centroid.

        This method calculates the distance from each data point to all centroids using
        the specified distance function (`self.distance_f`). It then assigns a label
        to each data point corresponding to the index of the nearest centroid.

        Attributes:
            self.X (list or ndarray): The dataset containing the data points.
            self.centroids (list or ndarray): The list of centroids.
            self.distance_f (callable): A function to compute the distance between a data point and a centroid.
            self.y_hat (list): A list to store the assigned labels for each data point.
            self.distances (list): A list to store the distances of each data point to all centroids.

        Effects:
            Updates `self.y_hat` with the index of the nearest centroid for each data point.
            Updates `self.distances` with the computed distances for each data point.
        """
        iter = 0
        self.y_hat=[]
        self.distances=[]
        for x in self.X:
            self.distances.append([self.distance_f(x,c) for c in self.centroids])
            self.y_hat.append(np.argmin(self.distances[iter]))
            iter+=1

    def plot_spread(self,xs = "Feature 1",ys = "Feature 2" ):
        """
        Visualizes the spread of data points and centroids in a 2D scatter plot.

        Parameters:
        -----------
        xs : str, optional
            Label for the x-axis of the plot. Default is "Feature 1".
        ys : str, optional
            Label for the y-axis of the plot. Default is "Feature 2".

        Returns:
        --------
        None
        """
        # Scatter plot of the data points, colored by their assigned labels
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y_hat, cmap='viridis', marker='o')
        
        # Scatter plot of the centroids
        centroids = np.array(self.centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        
        plt.title('K-Means Clustering')
        plt.xlabel(xs)
        plt.ylabel(ys)
        plt.legend()
        plt.show()

    def update_centroids(self):
        """
        Updates the centroids of the clusters based on the current assignments of data points.

        For each cluster, the new centroid is calculated as the mean of all data points
        assigned to that cluster.

        Attributes:
            self.centroids (list or np.ndarray): Current centroids of the clusters.
            self.X (list or np.ndarray): Dataset containing all data points.
            self.y_hat (list or np.ndarray): Cluster assignments for each data point.

        Updates:
            self.centroids (list or np.ndarray): Updated centroids after recalculating
            based on the current cluster assignments.
        """
        new_centroids = []
        for i in range(len(self.centroids)):
            points_in_cluster = [self.X[j] for j in range(len(self.X)) if self.y_hat[j] == i]
            if points_in_cluster:  # Avoid division by zero
                new_centroids.append(np.mean(points_in_cluster, axis=0))
            else:
                new_centroids.append(self.centroids[i])  # Keep the old centroid if no points are assigned
        self.centroids = new_centroids


    def update_loop(self, numLoops, solve=False, piter=True, maxIter=2000):
        """
        Performs the update loop for the k-means clustering algorithm.

        Parameters:
        -----------
        numLoops : int
            The number of iterations to perform if `solve` is False, or the 
            threshold for the stopping condition if `solve` is True.
        solve : bool, optional
            If True, the function will run until the centroids converge or 
            the maximum number of iterations (`maxIter`) is reached. If False, 
            the function will perform a fixed number of iterations (default is False).
        piter : bool, optional
            If True and `solve` is True, prints the number of iterations 
            performed before convergence (default is True).
        maxIter : int, optional
            The maximum number of iterations to perform when `solve` is True 
            (default is 2000).

        Behavior:
        ---------
        - If `solve` is True:
            - Iteratively updates centroids and assigns labels until the 
              change in centroids (delta) divided by the number of centroids 
              is less than `numLoops`, or the number of iterations reaches `maxIter`.
            - Prints the number of iterations if `piter` is True.
        - If `solve` is False:
            - Performs `numLoops` iterations of centroid updates and label assignments.
        """
        if solve==True:
            delta = 2000*numLoops
            iter = 0
            while (delta/len(self.centroids))>numLoops and iter<maxIter:
                old_centroids = self.centroids
                self.update_centroids()
                self.assign_label()
                delta = 0.0
                for cent_n,cent_o in zip(self.centroids,old_centroids):
                    delta += self.distance_f(cent_n,cent_o)
                iter += 1
            if piter==True:
                print(f"Number of iterations: {iter}")
        else:      
            for _ in range(numLoops):
                self.update_centroids()
                self.assign_label()
    