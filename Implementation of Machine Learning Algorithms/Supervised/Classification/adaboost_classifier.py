import pandas as pd
import numpy as np
import math
import statistics
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
from graphviz import Digraph, Source, Graph
from multiprocessing import cpu_count, Pool
import sklearn
from IPython.display import Math
from sklearn.tree import export_graphviz
from copy import deepcopy
from sklearn.metrics import pairwise_distances


class AdaboostClassifier():

    def __init__(self, n_estimators = 100, weights = None):
        self.n_estimators = n_estimators
        self.weights = weights
        self.alphas = []

    def _convert_y(self, y):
        self.actual_classes = sorted(np.unique(y))
        self.class_range = [-1, 1]
        self.class_range_to_actual_classes = dict(zip(*(self.class_range, self.actual_classes)))
        self.actual_classes_to_class_range = dict(zip(*(self.actual_classes, self.class_range)))

        y_ = np.array([self.actual_classes_to_class_range[i] for i in y])
        return y_

    def _get_true_class_labels(self, labels):
        true_labels = np.array([self.class_range_to_actual_classes[i] for i in labels])
        return true_labels

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        # Convert y to {-1, 1}
        y = self._convert_y(y)

        # Initialise weights for all data points
        row_length = X.shape[0]
        self.weights = np.ones((self.n_estimators, row_length))
        self.alphas = np.zeros((self.n_estimators, 1))
        self.estimators = np.empty((self.n_estimators, 1), dtype = object)

        time_step = 0
        for time_step in range(self.n_estimators):

            # Use a weak classifier to fit on data
            weak_classifier = LogisticRegression(solver = "sgd", epochs = 10)
            weak_classifier.fit(X, y)
            pred = weak_classifier.predict(X)

            # Get weighted error
            weighted_sample_err = (np.sum((pred != y) * self.weights))/np.sum(self.weights)

            # Alpha for current classifer
            alpha_t = 1/2*np.log(((1 - weighted_sample_err)/weighted_sample_err) + 1e-16)
            self.alphas[time_step] = alpha_t
            self.estimators[time_step] = weak_classifier

            # Update weights of next time step for all data points
            if time_step == (self.n_estimators - 1):
                break
            self.weights[time_step + 1, :] = self.weights[time_step, :] * np.exp(-y * alpha_t * pred)

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        self.estimators = self.estimators.flatten()
        self.alphas = self.alphas.flatten()
        for index in range(self.n_estimators):
            preds.append(self.alphas[index] * self.estimators[index].predict(X))

        preds = np.sum(preds, 0)
        preds = np.sign(preds)
        true_preds = self._get_true_class_labels(preds)
        return true_preds

    def get_accuracy(self, y, y_hat):
        return np.mean(y == y_hat)
