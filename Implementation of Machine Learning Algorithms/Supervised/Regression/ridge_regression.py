import pandas as pd
import numpy as np
import math
import statistics
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
import sklearn
from copy import deepcopy
from sklearn.metrics import pairwise_distances


class RidgeRegression():

    def __init__(self,
                 bias = None,
                 weights = None,
                 lambda_param = 10,
                 fit_intercept = True):
        self.bias = bias
        self.weights = weights
        self.fit_intercept = fit_intercept
        self.lambda_param = lambda_param

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))

        self.all_weights = np.linalg.inv(np.dot(X.T, X) + \
                            self.lambda_param * np.identity(X.shape[1])).dot(X.T).dot(y)
        self.weights = self.all_weights[1:]
        self.bias = self.all_weights[0]

    def predict(self, X):
        self.weights = self.weights.reshape(1, -1)
        predictions = self.bias + np.dot(self.weights, X.T)
        return predictions[0]

    def get_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)


# Load data
data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = RidgeRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_mse(y_pred, y_test)
print("Model Score = ", str(score))
