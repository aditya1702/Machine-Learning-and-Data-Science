import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


class LassoRegression():

    def __init__(self,
                 bias = None,
                 weights = None,
                 lambda_param = 10,
                 max_iters = 100,
                 fit_intercept = True):
        self.bias = 0
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.fit_intercept = fit_intercept

    def _soft_threshold(self, x, lambda_):
        if x > 0.0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0.0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0

    def fit(self, X, y):

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))

        row_length, column_length = X.shape

        # Define the weights
        self.weights = np.zeros((1, column_length))[0]
        if self.fit_intercept:
            self.weights[0] = np.sum(y - \
                                np.dot(X[:, 1:], self.weights[1:]))/(X.shape[0])

        #Looping until max number of iterations
        for iteration in range(self.max_iters):
            start = 1 if self.fit_intercept else 0

            #Looping through each coordinate
            for j in range(start, column_length):

                tmp_weights = self.weights.copy()
                tmp_weights[j] = 0.0
                r_j = y - np.dot(X, tmp_weights)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.lambda_param * X.shape[0]

                self.weights[j] = self._soft_threshold(arg1, arg2)/(X[:, j]**2).sum()

                if self.fit_intercept:
                    self.weights[0] = np.sum(y - \
                                        np.dot(X[:, 1:], self.weights[1:]))/(X.shape[0])

        self.bias = self.weights[0]
        self.weights = self.weights[1:]

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
model = LassoRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_mse(y_pred, y_test)
print("Model Score = ", str(score))
