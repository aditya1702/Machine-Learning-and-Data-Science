import pandas as pd
import numpy as np
import math
import statistics
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from graphviz import Digraph, Source, Graph
from IPython.display import Math
from sklearn.tree import export_graphviz


class Node():
    def __init__(self,
                 data = None,
                 split_variable = None,
                 split_variable_value = None,
                 left = None,
                 right = None,
                 depth = 0,
                 criterion_value = None):
        self.data = data
        self.split_variable = split_variable
        self.split_variable_value = split_variable_value
        self.left = left
        self.right = right
        self.criterion_value = criterion_value
        self.depth = depth

class DecisionTreeRegressor():
    def __init__(self,
                 root = None,
                 criterion = "mse",
                 max_depth = 2,
                 significance = None,
                 significance_threshold = 3.841,
                 min_samples_split = 10):
        self.root = root
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.significance = significance
        self.significance_threshold = significance_threshold

        self.split_score_funcs = {'mse': self._calculate_mse_values}

    def _get_mse(self, X):
        if X.empty:
            return 0

        # Calculate the mean square error with respect to the mean
        y = X['Y']
        y_mean = np.mean(y)
        mse = np.mean((y - y_mean)**2)
        return mse

    def _calculate_mse_values(self, X, feature):

        # Calculate unique values of X. For a feature, there are different
        # values on which that feature can be split
        classes = X[feature].unique()

        # Calculate the gini value for a split on each unique value of the feature.
        best_mse_score = np.iinfo(np.int32(10)).max
        best_feature_value = ""
        for unique_value in classes:
            # Split data
            left_split = X[X[feature] <= unique_value]
            right_split = X[X[feature] > unique_value]

            # Get gini scores of left, right nodes
            mse_value_left_split = self._get_mse(left_split)
            mse_value_right_split = self._get_mse(right_split)

            # Combine the 2 scores to get the overall score for the split
            mse_score_of_current_value = (left_split.shape[0]/X.shape[0]) * mse_value_left_split + \
                                           (right_split.shape[0]/X.shape[0]) * mse_value_right_split
            if mse_score_of_current_value < best_mse_score:
                best_mse_score = mse_score_of_current_value
                best_feature_value = unique_value

        return best_mse_score, best_feature_value

    def _get_best_split_feature(self, X):
        best_split_score = np.iinfo(np.int32(10)).max
        best_feature = ""
        best_value = None
        columns = X.drop('Y', 1).columns

        for feature in columns:

            # Calculate the best split score and the best value
            # for the current feature.
            split_score, feature_value = self.split_score_funcs[self.criterion](X, feature)

            # Compare this feature's split score with the current best score
            if split_score < best_split_score:
                best_split_score = split_score
                best_feature = feature
                best_value = feature_value

        return best_feature, best_value, best_split_score

    def _split_data(self, X, X_depth = None):

        # Return if dataframe is empty, depth exceeds maximum depth or sample size exceeds
        # minimum sample size required to split.
        if X.empty or len(X['Y'].value_counts()) == 1 or X_depth == self.max_depth \
                            or X.shape[0] <= self.min_samples_split:
            return None, None, "", "", 0

        # Calculate the best feature to split X
        best_feature, best_value, best_score = self._get_best_split_feature(X)

        if best_feature == "":
            return None, None, "", "", 0

        # Create left and right nodes
        X_left = Node(data = X[X[best_feature] <= best_value].drop(best_feature, 1),
                      depth = X_depth + 1)
        X_right = Node(data = X[X[best_feature] > best_value].drop(best_feature, 1),
                       depth = X_depth + 1)

        return X_left, X_right, best_feature, best_value, best_score

    def _fit(self, X):

        # Handle the initial case
        if not (type(X) == Node):
            X = Node(data = X)
            self.root = X

        # Get the splits
        X_left, X_right, best_feature, best_value, best_score = self._split_data(X.data, X.depth)

        # Assign attributes of node X
        X.left = X_left
        X.right = X_right
        X.split_variable = best_feature
        X.split_variable_value = round(best_value, 3) if type(best_value) != str else best_value
        X.criterion_value = round(best_score, 3)

        # Return if no best variable found to split on.
        # This means you have reached the leaf node.
        if best_feature == "":
            return

        # Recurse for left and right children
        self._fit(X_left)
        self._fit(X_right)

    def fit(self, X, y):

        # Combine the 2 and fit
        X = pd.DataFrame(X)
        X['Y'] = y
        self._fit(X)

    def predict(self, X):
        X = np.asarray(X)
        X = pd.DataFrame(X)

        preds = []
        for index, row in X.iterrows():
            curr_node = self.root
            while(curr_node.left != None and curr_node.right != None):
                split_variable = curr_node.split_variable
                split_variable_value = curr_node.split_variable_value
                if X.loc[index, split_variable] <= split_variable_value:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right

            # Get prediction
            preds.append(np.mean(curr_node.data['Y'].values))

        return preds

    def display_tree_structure(self):
        tree = Digraph('DecisionTree',
                       filename = 'tree.dot',
                       node_attr = {'shape': 'box'})
        tree.attr(size = '10, 20')
        root = self.root
        id = 0

        # queue with nodes to process
        nodes = [(None, root, 'root')]
        while nodes:
            parent, node, x = nodes.pop(0)

            # Generate appropriate labels for the nodes
            value_counts_length = len(node.data['Y'].value_counts())
            if node.split_variable != "":
                split_variable = node.split_variable
                split_variable_value = node.split_variable_value
            else:
                split_variable = "None"

            if value_counts_length > 1:
                label = str(split_variable) + '\n' + str(self.criterion) + " = " + \
                            str(node.criterion_value)
            else:
                label = "None"

            # Make edges between the nodes
            tree.node(name = str(id),
                      label = label,
                      color = 'black',
                      fillcolor = 'goldenrod2',
                      style = 'filled')

            if parent is not None:
                if x == 'left':
                    tree.edge(parent, str(id), color = 'sienna',
                              style = 'filled', label = '<=' + ' ' + str(split_variable_value))
                else:
                    tree.edge(parent, str(id), color = 'sienna',
                              style = 'filled', label = '>' + ' ' + str(split_variable_value))

            if node.left is not None:
                nodes.append((str(id), node.left, 'left'))

            if node.right is not None:
                nodes.append((str(id), node.right, 'right'))
            id += 1

        return tree

    def get_error(self, y, y_hat):
        return np.mean((y - y_hat)**2)


# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fit model
model = DecisionTreeRegressor(max_depth = 3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Get accuracy
score = model.get_accuracy(y_pred, y_test)
print("Model Score = ", str(score))
