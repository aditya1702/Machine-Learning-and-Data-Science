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


class NeuralNetworkClassifier():

    SigmoidActivation = "sigmoid"
    ReLUActivation = "relu"
    LinearActivation = "linear"
    SoftmaxActivation = "softmax"

    def __init__(self,
                 num_hidden_layers = 2,
                 learning_rate = 0.1,
                 num_neurons_each_layer = None,
                 num_neurons_last_layer = 1,
                 batch_size = 32,
                 epochs = 10,
                 weights = None):
        self.weights = weights
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neurons_last_layer = num_neurons_last_layer

        # Sigmoid activation for other layers. Softmax activation for last layer
        self.activations = [self.SigmoidActivation] * self.num_hidden_layers + [self.SoftmaxActivation]
        self.activations_functions = {
            self.SigmoidActivation: self._sigmoid,
            self.ReLUActivation: self._relu,
            self.LinearActivation: self._linear,
            self.SoftmaxActivation: self._softmax
        }
        self.activations_derivatives = {
            self.SigmoidActivation: self._sigmoid_derivative,
            self.ReLUActivation: self._relu_derivative,
            self.LinearActivation: self._linear_derivative,
            self.SoftmaxActivation: self._softmax_derivative,
        }

    def _sigmoid(self, x):
        def sigfunc(x):
            if x < 0:
                return 1 - 1 / (1 + math.exp(x))
            else:
                return 1 / (1 + math.exp(-x))
        x_ = np.array([sigfunc(i) for i in x])
        return x_

    def _softmax(self, z):
        e_x = np.exp(z)
        out = e_x / e_x.sum(axis = 0)
        return out

    def _relu(self, x):
        return np.maximum(0, x)

    def _linear(self, x):
        return x

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _softmax_derivative(self, x):
        e_x = np.exp(x)
        e_x_sum = np.sum(e_x)
        out = e_x * (e_x_sum - e_x)/(e_x_sum)**2
        return out

    def _relu_derivative(self, x):
        return (np.ones_like(x) * (x > 0))

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _binary_cross_entropy_loss(self, y_hat, y):
        return np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

    def _initialise_weights(self, input_shape):

        self.num_neurons_each_layer.append(self.num_neurons_last_layer)
        self.total_layers = self.num_hidden_layers + 1
        self.layers = range(self.total_layers)

        # Initialising a numpy array of
        # shape = (number of hidden layers, number of neurons, number of weights per neuron) to store weights
        self.weights = []

        # Iterate through the layers
        for layer in self.layers:
            self.weights.append([])

            number_of_neurons_in_this_layer = self.num_neurons_each_layer[layer]
            if layer == 0:
                fan_in = input_shape
                fan_out = number_of_neurons_in_this_layer
                previous_layer_shape = fan_in
            else:
                fan_in = self.num_neurons_each_layer[layer - 1]
                fan_out = number_of_neurons_in_this_layer
                previous_layer_shape =  1 + fan_in

            init_bound = np.sqrt(6. / (fan_in + fan_out))
            self.weights[layer] = np.random.uniform(low = -init_bound,
                                                    high = init_bound,
                                                    size = (number_of_neurons_in_this_layer,
                                                           previous_layer_shape))

        self.weights = np.array(self.weights)
        self.old_weights = deepcopy(self.weights)

    def _update_weights(self):
        avg_batch_weight_derivatives = np.mean(self.batch_weight_derivatives, axis = 0)
        self.weights = self.old_weights - self.learning_rate * avg_batch_weight_derivatives
        self.old_weights = deepcopy(self.weights)
        self.batch_weight_derivatives = []

    def _convert_to_indicator(self, y):
        y_indicator = np.zeros((y.shape[0], self.num_classes))
        for index, y_value in enumerate(y):
            class_range_mapping = int(self.actual_classes_to_class_range[y_value])
            y_indicator[index, class_range_mapping] = 1
        return y_indicator

    def _backward(self, x, y, out):

        # The derivatives array will have the same shape as weights array. - one derivative for each
        # weight
        output_derivatives = deepcopy(out)
        weight_derivatives = deepcopy(self.weights)

        # Compute the output derivatives
        layers_reversed = self.layers[::-1]
        for curr_layer in layers_reversed:
            next_layer = curr_layer + 1

            # For the last layer simply use the formula
            if curr_layer == self.total_layers - 1:
                output_derivatives[curr_layer] = -y/(out[curr_layer] + 1e-16) + \
                                                (1 - y) * 1/(1 - out[curr_layer] + 1e-16)
                continue

            # Get the activation derivative function for next layer
            activation_for_next_layer = self.activations[next_layer]
            activation_derivative = self.activations_derivatives[activation_for_next_layer]

            # The next layer output derivatives
            next_layer_output_derivatives = output_derivatives[next_layer]

            # Calculate the activation derivative. Add a 1 for the bias weight
            current_layer_output = out[curr_layer].copy()
            current_layer_output = np.insert(current_layer_output, obj = 0, values = 1)
            next_layer_activation_derivatives = activation_derivative(self.old_weights[next_layer] @ current_layer_output)
            next_layer_activation_derivatives = next_layer_activation_derivatives.reshape(-1, 1)

            # Remove the bias from the weights.
            next_layer_weights_without_bias = self.old_weights[next_layer][:, 1:]

            # Multiply each neuron's activation derivative with its weights. This is the Hadmard product
            second_term = next_layer_activation_derivatives * next_layer_weights_without_bias

            # Sum over all the neurons in the next layer to get the output derivative for each
            # neuron in the current layer. This is because each neuron contributes to all the neurons
            # in the next layer.
            output_derivatives[curr_layer] = next_layer_output_derivatives @ second_term

        # Update the weights using the output derivative calculated above
        for curr_layer in layers_reversed:

            # Get the activation for this layer and its derivative function
            activation_for_this_layer = self.activations[curr_layer]
            activation_derivative = self.activations_derivatives[activation_for_this_layer]

            # If first layer then use the data as the previous layer.
            if curr_layer == 0:
                previous_layer_output = x
            else:
                prev_layer = curr_layer - 1
                previous_layer_output = out[prev_layer].copy()
                previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1)

            # Current layer output derivatives
            curr_layer_output_derivatives = output_derivatives[curr_layer].reshape(-1, 1)

            # Get current layer's activation derivatives
            curr_layer_activation_derivatives = activation_derivative(self.old_weights[curr_layer] @ previous_layer_output)
            curr_layer_activation_derivatives = curr_layer_activation_derivatives.reshape(-1, 1)

            # For the current layer multiply each neuron's activation derivatives with all previous layer outputs.
            curr_layer_weight_derivatives = curr_layer_output_derivatives * \
                                            curr_layer_activation_derivatives * previous_layer_output
            weight_derivatives[curr_layer] = curr_layer_weight_derivatives

        # Append the current data point's weight derivatives in the batch derivatives array
        self.batch_weight_derivatives.append(weight_derivatives)

    def _forward(self, x):
        out = []
        for curr_layer in self.layers:
            out.append([])

            # Get the activation for this layer and its function
            activation_for_this_layer = self.activations[curr_layer]
            activation_function = self.activations_functions[activation_for_this_layer]

            if curr_layer == 0:
                previous_layer_output = x
            else:
                previous_layer_output = out[curr_layer - 1].copy()
                previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1)

            out[curr_layer] = activation_function(self.weights[curr_layer] @ previous_layer_output)

        out = np.array(out)
        return out

    def fit(self, X, y):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))

        # Initialise the weights of the network
        self._initialise_weights(X_new.shape[1])

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

        for epoch in range(self.epochs):

            # Initialise arrays to store all weight derivatives of the batch
            self.batch_weight_derivatives = []

            # Update weights using mini-batch stochastic gradient descent
            for data_index in range(X_new.shape[0]):
                out = self._forward(X_new[data_index])
                self._backward(X_new[data_index], y[data_index], out)

                if (data_index + 1) % self.batch_size == 0:
                    self._update_weights()

            predictions, probs = self.predict(X)
            loss = self._binary_cross_entropy_loss(probs, y)
            print("Epoch = ", str(epoch + 1), " - ", "Loss = ", str(loss))

    def predict(self, X):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))

        preds = []
        preds_probs = []
        for x in X_new:
            last_layer_out = self._forward(x)[-1]
            preds.append(np.argmax(last_layer_out))
            preds_probs.append(last_layer_out)

        preds = np.array(preds)
        preds_probs = np.array(preds_probs)
        return preds, preds_probs
