import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances


class KMeans():

    def __init__(self, k = 5, max_iters = 100, random_seed = 42):
        self.k = k
        self.max_iters = max_iters

        # Set random seed
        np.random.seed(random_seed)

    def _initialise_centroids(self, X):
        random_indices = np.random.permutation(X.shape[0])
        random_indices = random_indices[:self.k]
        self.centroids = X[random_indices]

    def _euclidien_distance(self, x):
        return np.sum((x - self.centroids)**2, axis = 1)

    def _assign_clusters(self, X):
        cluster_distances = pairwise_distances(X, self.centroids, metric = 'euclidean')
        cluster_labels = np.argmin(cluster_distances, axis = 1)
        return cluster_labels

    def _update_centroids(self, X, cluster_labels):
        for cluster in range(self.k):

            # Get all data points of a cluster
            X_cluster = X[cluster_labels == cluster]

            # Update the cluster's centroid
            cluster_mean = np.mean(X_cluster, axis = 0)
            self.centroids[cluster] = cluster_mean

    def fit(self, X):

        # Initialise random centroids
        self._initialise_centroids(X)

        iterations = 0
        while iterations <= self.max_iters:
            iterations += 1

            # Assign clusters to data
            cluster_labels = self._assign_clusters(X)

            # Update centroids
            self._update_centroids(X, cluster_labels)

    def predict(self, X):
        return self._assign_clusters(X)


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test = train_test_split(X, test_size = 0.1)

# Fit model
model = KMeans(k = 5)
model.fit(X_train)

# Predict
y_pred = model.predict(X_test)
print(y_pred)
