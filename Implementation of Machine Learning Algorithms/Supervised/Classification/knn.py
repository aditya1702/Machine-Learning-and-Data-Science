import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


class KNeighbours():
    def __init__(self, k = 5, distance_metric = 'euclid', problem = "classify"):
        self.k = k
        self.distance_metric = distance_metric
        self.problem = problem
        self.prediction_functions = {'classify': self._top_k_votes,
                                     'regress': self._top_k_mean}
        self.eval_functions = {'classify': self._get_accuracy,
                               'regress': self._get_mse}

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def _euclidien_distance(self, x):
        return np.sqrt(np.sum((x - self.X)**2, axis = 1))

    def _top_k_mean(self, top_k):
        return np.mean(top_k)

    def _top_k_votes(self, top_k):
        return max(top_k, key = list(top_k).count)

    def predict(self, X):
        preds = list()
        X = np.asarray(X)
        for x in X:
            distances = self._euclidien_distance(x)

            # Zip the distances and y values together
            distances = zip(*(distances, self.y))

            # Sort the distances list by distance values in descending order
            distances = sorted(distances, key = lambda x: x[0])

            # Select top k distances
            top_k = distances[:(self.k)]

            top_k = np.array(top_k)
            top_k = top_k[:, 1]

            # Calculate mean of y values of these top k data items
            pred = self.prediction_functions[self.problem](top_k)
            preds.append(pred)

        return preds

    def evaluate(self, pred, y):
        eval_func = self.eval_functions[self.problem]
        return eval_func(pred, y)

    def _get_accuracy(self, pred, y):
        return np.mean(pred == y)*100

    def _get_mse(self, pred, y):
        return np.mean((pred - y)**2)


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = KNeighbours(problem = "classify")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.evaluate(y_pred, y_test)
print("Model Score = ", str(score))
