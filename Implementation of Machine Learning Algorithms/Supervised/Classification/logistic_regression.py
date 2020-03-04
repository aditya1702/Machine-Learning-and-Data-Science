import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


class LogisticRegression():

    def __init__(self,
                 weights = None,
                 bias = None,
                 fit_intercept = True,
                 decision_threshold = 0.5,
                 epochs = 50,
                 solver = 'sgd',
                 batch_size = 30,
                 learning_rate = 0.05,
                 tolerance = 1e-13):
        self.weights = weights
        self.bias = bias
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.decision_threshold = decision_threshold
        self.epochs = epochs
        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.solver_func = {'newton': self._newton_solver,
                            'sgd': self._sgd_solver}

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _log_likelihood(self, X, y):
        P = self._sigmoid(X @ self.weights)
        P = P.reshape(-1, 1)
        log_P = np.log(P + 1e-16)

        P_ = 1 - P
        log_P_ = np.log(P_ + 1e-16)
        log_likelihood = np.sum(y*log_P + (1 - y)*log_P_)
        return log_likelihood

    def _get_true_class_labels(self, labels):
        true_labels = np.array([self.class_range_to_actual_classes[i] for i in labels])
        return true_labels

    def _get_batches(self, X, y):
        for i in range(0, X.shape[0], self.batch_size):
            yield (X[i: i + self.batch_size], y[i: i + self.batch_size])

    def _convert_y(self, y):
        self.actual_classes = sorted(np.unique(y))
        self.class_range = [0, 1]
        self.class_range_to_actual_classes = dict(zip(*(self.class_range, self.actual_classes)))
        self.actual_classes_to_class_range = dict(zip(*(self.actual_classes, self.class_range)))

        y_ = np.array([self.actual_classes_to_class_range[i] for i in y])
        y_ = y_.reshape(-1, 1)
        return y_

    def _newton_solver(self, X, y):
        log_likelihood = self._log_likelihood(X, y)
        iterations = 0
        delta = np.inf
        while(np.abs(delta) > self.tolerance and iterations < self.epochs):
            iterations += 1

            # Calculate positive class probabilities: p = sigmoid(W*x + b)
            z = X @ self.weights
            P = self._sigmoid(z)
            P = P.reshape(-1, 1)

            # First derivative of loss w.r.t weights
            grad = X.T @ (P - y)

            # Hessian of loss w.r.t weights
            P_ = 1 - P
            W = P * P_
            W = W.reshape(1, -1)[0]
            W = np.diag(W)
            hess = X.T @ W @ X

            # Weight update using Newton-Rhapson Method
            self.weights -= np.linalg.inv(hess) @ grad

            # Calculate new log likelihood
            new_log_likelihood = self._log_likelihood(X, y)
            delta = log_likelihood - new_log_likelihood
            log_likelihood = new_log_likelihood

    def _sgd_solver(self, X, y):
        iterations = 0
        while(iterations < self.epochs):
            iterations += 1

            # Get batches
            batches = self._get_batches(X, y)

            # Update weights using Mini batch stochastic gradient descent
            for (x_batch, y_batch) in batches:

                # Raw output
                z = x_batch @ self.weights

                # Calculate positive class probabilities: p = sigmoid(W*x + b)
                P = self._sigmoid(z)

                # First derivative of loss w.r.t weights
                grad = x_batch.T @ (P - y_batch)

                # Update weights
                self.weights -= self.learning_rate * grad

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))
        row_length, column_length = X.shape

        # Define the weights
        self.weights = np.zeros((column_length, 1))

        # Convert y to {0, 1}
        y = self._convert_y(y)

        # Use the solver
        self.solver_func[self.solver](X, y)

    def predict_proba(self, X):

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))
        else:
            X = np.column_stack((np.zeros(len(X)), X))

        z = X @ self.weights
        predicted_probs = self._sigmoid(z)
        return predicted_probs

    def predict(self, X):
        predict_probs = self.predict_proba(X)
        preds = np.where(predict_probs < 0.5, 0, 1).flatten()
        true_preds = self._get_true_class_labels(preds)
        return true_preds

    def get_accuracy(self, y, y_hat):
        return np.mean(y == y_hat)*100


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = LogisticRegression(solver = "sgd", epochs = 100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_accuracy(y_pred, y_test)
print("Model Score = ", str(score))
