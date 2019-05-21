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


class Perceptron():

    def __init__(self, w = None, b = None, termination_steps = 0):
        self.w = w
        self.b = b
        self.termination_steps = termination_steps
        self.prev_weights = []
        self.prev_bias = []

    def fit(self, X, y):
        self.w = np.zeros((1, X.shape[1]))
        self.b = 0

        while(True):

            # Update w, b by iterating over each row of X
            misclassified_count = 0
            for index in range(len(y)):

                # Update when a data point gets misclassified
                pred = np.sign(np.dot(self.w, X[index]) + self.b)[0]
                if pred != y[index]:
                    misclassified_count += 1
                    self.w += X[index]*y[index]
                    self.b += y[index]

            # Termination condition
            if (misclassified_count == 0) or (misclassified_count >= 0.3*len(y) \
                                              and self.termination_steps >= 1e5):
                break

            self.prev_weights.append(self.w.copy())
            self.prev_bias.append(copy(self.b))
            self.termination_steps += 1

    def predict(self, X):
        return np.sign(self.w @ X.T + self.b)[0]

    def get_accuracy(self, y, y_hat):
        return sum(y == y_hat)*100/len(y)
