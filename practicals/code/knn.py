import numpy as np
import scipy as sp

class KNN:
    def __init__(self, X, y, k, trainsplit):
        self.k = k
        self.X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # normalize features
        self.y = y
        self.trainsplit = trainsplit
        
    def test_train_split(self):
        """Splits the dataset into training and testing sets based on the trainsplit ratio."""
        if self.trainsplit < 1.0:
            split_index = int(self.trainsplit * self.X.shape[0])
            self.X_train = self.X[:split_index, :]
            self.y_train = self.y[:split_index, :]
            self.X_test = self.X[split_index:, :]
            self.y_test = self.y[split_index:, :]
        else:
            raise ValueError("trainsplit should be a float between 0 and 1.")
    
    def eucl_dist_matrix(self, X_test, X_train):
        """
        Computes the Euclidean distance matrix between two sets of vectors A and B. The result
        should be a matrix of shape (A.shape[0], B.shape[0]).
        """
        dists = sp.spatial.distance.cdist(X_test, X_train, 'euclidean')
        shape = dists.shape
        if shape[0] != X_test.shape[0] or shape[1] != X_train.shape[0]:
            raise ValueError("The shape of the distance matrix is incorrect.")
        return dists
    
def test_function():
    print("This is a test function from knn.py")
    

    