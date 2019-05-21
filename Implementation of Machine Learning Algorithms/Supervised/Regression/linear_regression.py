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


class LinearRegression():

    def __init__(self, bias = None, weights = None, fit_intercept = True):
        self.bias = bias
        self.weights = weights
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        self.all_weights = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
        self.weights = self.all_weights[1:]
        self.bias = self.all_weights[0]

    def predict(self, X):
        self.weights = self.weights.reshape(1, -1)
        predictions = self.bias + np.dot(self.weights, X.T)
        return predictions[0]

    def get_mse(self, y_true, y_pred):
        return np.sum((y_true - y_pred)**2)/y_true.shape[0]
