import numpy as np
import matplotlib.pyplot as plt


class kmeans(object):
    def __init__(self,x,y,distance):
        # distance must be a function that takes in a value x and the self.centroids matrix and give the distnaces between x and each row of the matrix 
        self.X = x
        self.y = y
        self.centroids = []
        self.distance_f = distance
        self.distances = []
        self.y_hat = []

    def make_centroids(self,k):
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
        iter = 0
        self.y_hat=[]
        self.distances=[]
        for x in self.X:
            self.distances.append([self.distance_f(x,c) for c in self.centroids])
            self.y_hat.append(np.argmin(self.distances[iter]))
            iter+=1

    def plot_spread(self,xs = "Feature 1",ys = "Feature 2" ):
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
        new_centroids = []
        for i in range(len(self.centroids)):
            points_in_cluster = [self.X[j] for j in range(len(self.X)) if self.y_hat[j] == i]
            if points_in_cluster:  # Avoid division by zero
                new_centroids.append(np.mean(points_in_cluster, axis=0))
            else:
                new_centroids.append(self.centroids[i])  # Keep the old centroid if no points are assigned
        self.centroids = new_centroids


    def update_loop(self,numLoops):
        for _ in range(numLoops):
            self.update_centroids()
            self.assign_label()
    