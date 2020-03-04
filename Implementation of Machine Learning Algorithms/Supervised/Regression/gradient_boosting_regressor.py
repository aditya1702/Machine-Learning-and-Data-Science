import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.tree._tree import TREE_LEAF
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor as sklearn_gbm
np.seterr(all = "ignore")


class InitialEstimator():

    def __init__(self):
        return

    def fit(self, X, y):
        self.y = y

    def predict(self, X):
        preds = [np.mean(self.y)]*(X.shape[0])
        preds = np.array(preds)
        return preds

class GradientBoostingRegressor():

    def __init__(self, n_estimators = 100, max_depth = 5, loss = "mse", learning_rate = 0.01):
        self.estimators = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.loss = loss
        self.learning_rate = learning_rate

        self.residual_functions = {
            'mse': self._mse_residual,
            'mae': self._mae_residual
        }

        self.update_functions = {
            'mse': self._update_mse_terminal_regions,
            'mae': self._update_mae_terminal_regions
        }

    def _update_mse_terminal_regions(self, tree, X, raw_predictions):
        # For mse, we just update the raw predictions.
        # F(m) = F(m-1) + eta * delta(m)
        raw_predictions += self.learning_rate * tree.predict(X).ravel()
        return raw_predictions

    def _update_mae_terminal_regions(self, tree, X, raw_predictions):

        # Get the tree leaves
        terminal_regions = tree.apply(X)

        # update each leaf
        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            terminal_region = np.where(terminal_regions == leaf)[0]
            diff = (y.take(terminal_region, axis = 0) - raw_predictions.take(terminal_region, axis = 0))
            tree.value[leaf, 0, 0] = np.median(diff)

        # update predictions (both in-bag and out-of-bag)
        raw_predictions += self.learning_rate * tree.value[:, 0, 0].take(terminal_regions, axis = 0)
        return raw_predictions


    def _mse_residual(self, y, raw_predictions):
        return y - raw_predictions.ravel()

    def _mae_residual(self, y, raw_predictions):
        raw_predictions = raw_predictions.ravel()
        return 2 * (y - raw_predictions > 0) - 1

    def fit(self, X, y):

        X = X.astype(np.float32)
        n_samples = X.shape[0]

        # Fit an initial estimator at step 0
        init_estimator = InitialEstimator()
        init_estimator.fit(X, y)
        raw_predictions = init_estimator.predict(X)
        self.estimators.append(init_estimator)

        # Boosting iterations
        for step in range(self.n_estimators):

            # Calculate the pseudo-residuals (negative gradient)
            residuals = self.residual_functions[self.loss](y, raw_predictions.copy())

            # Fit a weak regression tree model to the residuals
            tree = DecisionTreeRegressor(max_depth = self.max_depth)
            tree.fit(X, residuals)

            # Update tree leaves (line search)
            raw_predictions = self.update_functions[self.loss](tree.tree_, X, raw_predictions)

            # Add the tree to ensemble
            self.estimators.append(tree)

    def predict(self, X):
        predictions = self.estimators[0].predict(X) + \
                        self.learning_rate * np.sum([model.predict(X) for model in self.estimators[1:]], 0)
        return predictions

    def get_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)


# Load data
data = load_boston()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = GradientBoostingRegressor(n_estimators = 50, loss = 'mse', max_depth = 3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_mse(y_pred, y_test)
print("Model Score = ", str(score))
