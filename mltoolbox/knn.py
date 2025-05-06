import numpy as np


class KNN(object):
    def __init__(self, X_train, y_train, distance, k=5):
        self.x = X_train
        self.y = y_train
        self.distance = distance
        self.k = k

    def k_nearest_neighbors(self, point, k):
        # Create an empty list to store neighbors and distances
        neighbors = []
        
        for p, label in zip(self.x, self.y):
            d = self.distance(point, p)
            temp_data = [p, label, d]
            neighbors.append(temp_data)
            
        neighbors.sort(key = lambda x : x[-1])
        
        return neighbors[:k]
    
    def KNN_Predict(self, point, k, regression = False):

        neighbors = self.k_nearest_neighbors(point, k)
        
        if regression == False:
            labels = [x[1] for x in neighbors]
            return max(labels, key = labels.count)
        
        else:
            return sum(x[1] for x in neighbors)/k

        
    def classification_error(self, test_features, test_labels, k):
        error = 0
        for point, label in zip(test_features, test_labels):
            error += label != self.KNN_Predict(point, k)
        return error/len(test_features)
    

