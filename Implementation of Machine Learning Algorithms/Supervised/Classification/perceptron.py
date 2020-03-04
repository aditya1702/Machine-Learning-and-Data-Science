import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


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
            self.prev_bias.append(self.b.copy())
            self.termination_steps += 1

    def predict(self, X):
        return np.sign(self.w @ X.T + self.b)[0]

    def get_accuracy(self, y, y_hat):
        return np.mean(y == y_hat)*100


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = Perceptron()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_accuracy(y_pred, y_test)
print("Model Score = ", str(score))
