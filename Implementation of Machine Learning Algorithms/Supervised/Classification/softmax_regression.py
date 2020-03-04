import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


class MultiClassLogisticRegression():

    def __init__(self,
                 weights = None,
                 bias = None,
                 fit_intercept = True,
                 epochs = 50,
                 learning_rate = 0.05,
                 batch_size = 50):
        self.weights = weights
        self.learning_rate = learning_rate
        self.bias = bias
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.batch_size = batch_size

    def _softmax(self, z):

        # We only calculate the softmax probabilities of the first (K-1) classes
        z_ = z[:, :(z.shape[1] - 1)]
        e_x = np.exp(z_)
        out_k_minus_1 = e_x / (1 + e_x.sum(axis = 1, keepdims = True))

        # Probability for last K = 1 - p((K - 1))
        out_k = 1 - out_k_minus_1.sum(axis = 1)
        out = np.column_stack((out_k_minus_1, out_k))

        return out

    def _get_true_class_labels(self, P):
        labels = P.argmax(axis = 1)
        labels = np.array([self.class_range_to_actual_classes[i] for i in labels])
        return labels

    def _calculate_cross_entropy(self, y, log_yhat):
        return -np.sum(y * log_yhat, axis = 1)

    def _convert_to_indicator(self, y):
        y_indicator = np.zeros((y.shape[0], self.num_classes))
        for index, y_value in enumerate(y):
            class_range_mapping = int(self.actual_classes_to_class_range[y_value])
            y_indicator[index, class_range_mapping] = 1
        return y_indicator

    def _get_batches(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            yield (X[i: i + self.batch_size], y[i: i + self.batch_size])

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        row_length, column_length = X.shape

        # Number of unique classes
        self.actual_classes = sorted(np.unique(y))
        self.num_classes = len(self.actual_classes)

        # This will generate a list of [0,1,2,3....]. However, we want to map these class labels
        # to the original class labels in Y
        self.class_range = list(range(self.num_classes))
        self.class_range_to_actual_classes = dict(zip(*(self.class_range, self.actual_classes)))
        self.actual_classes_to_class_range = dict(zip(*(self.actual_classes, self.class_range)))

        # Convert y to indicator matrix form e.g. If y belongs to class 3, then y = [0,0,1,0..0]
        y = self._convert_to_indicator(y)

        # Define the weights, shape = (P + 1, K)
        self.weights = np.zeros((column_length, self.num_classes))

        iterations = 0
        while(iterations < self.epochs):
            iterations += 1

            # Get batches
            batches = self._get_batches(X, y)

            # Update weights using Mini batch stochastic gradient descent
            for (x_batch, y_batch) in batches:

                # Get raw output
                z = x_batch @ self.weights

                # Calculate class probabilities from raw output, shape = (B, K); B = batch size
                P = self._softmax(z)

                # Calculate gradient
                grad = x_batch.T @ (P - y_batch)

                # Update weights
                self.weights -= self.learning_rate * grad

    def predict_proba(self, X):

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))

        z = X @ self.weights
        predicted_probs = self._softmax(z)
        return predicted_probs

    def predict(self, X):
        predicted_probs = self.predict_proba(X)
        preds = self._get_true_class_labels(predicted_probs)
        return preds

    def get_accuracy(self, y, y_hat):
        return np.mean(y == y_hat)*100


# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = MultiClassLogisticRegression(epochs = 100, learning_rate = 0.05, batch_size = 100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_accuracy(y_pred, y_test)
print("Model Score = ", str(score))
