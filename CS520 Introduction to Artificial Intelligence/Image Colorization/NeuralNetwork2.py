import numpy as np
import math
from copy import deepcopy

np.seterr(all='raise')
class NeuralNetwork():

    SigmoidActivation = "sigmoid"
    ReLUActivation = "relu"
    LinearActivation = "linear"

    def __init__(self,
                 num_hidden_layers = 2,
                 learning_rate = 0.1,
                 num_neurons_each_layer = None,
                 batch_size = 32,
                 epochs = 10,
                 weights = None):
        self.weights = weights
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Sigmoid activation for other layers. Linear activation for last layer
        self.activations = [self.SigmoidActivation] * self.num_hidden_layers + [self.LinearActivation]
        self.activations_functions = {
            self.SigmoidActivation: self._sigmoid,
            self.ReLUActivation: self._relu,
            self.LinearActivation: self._linear
        }
        self.activations_derivatives = {
            self.SigmoidActivation: self._sigmoid_derivative,
            self.ReLUActivation: self._relu_derivative,
            self.LinearActivation: self._linear_derivative
        }

    def _sigmoid(self, x):
        def sigfunc(x):
            if x < 0:
                return 1 - 1 / (1 + math.exp(x))
            else:
                return 1 / (1 + math.exp(-x))
        x_ = np.array([sigfunc(i) for i in x])
        return x_

    def _relu(self, x):
        return np.maximum(0, x)

    def _linear(self, x):
        return x

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu_derivative(self, x):
        return (np.ones_like(x) * (x > 0))

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _mse_loss(self, pred, y):
        return np.mean((pred - y) ** 2)

    def _initialise_weights(self, input_shape):

        self.num_neurons_each_layer.append(1)
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
                self.weights[layer] = np.random.normal(loc = 0,
                                                       scale = 0.5,
                                                       size = (number_of_neurons_in_this_layer, input_shape))
            else:
                # Adding 1 for the bias neuron
                self.weights[layer] = np.random.normal(loc = 0,
                                                       scale = 0.5,
                                                       size = (number_of_neurons_in_this_layer,
                                                     1 + self.num_neurons_each_layer[layer - 1]))

        self.weights = np.array(self.weights)
        self.old_weights = deepcopy(self.weights)

    def _update_weights(self):
        avg_batch_weight_derivatives = np.mean(self.batch_weight_derivatives, axis = 0)
        self.weights = self.old_weights - self.learning_rate * avg_batch_weight_derivatives
        self.old_weights = deepcopy(self.weights)
        self.batch_weight_derivatives = []

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
                output_derivatives[curr_layer] = 2*(out[curr_layer] - y)
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

        for epoch in range(self.epochs):

            # Initialise arrays to store all weight derivatives of the batch
            self.batch_weight_derivatives = []

            # Update weights using mini-batch gradient descent
            for data_index in range(X_new.shape[0]):
                out = self._forward(X_new[data_index])
                self._backward(X_new[data_index], y[data_index], out)

                if (data_index + 1) % self.batch_size == 0:
                    self._update_weights()

            predictions = self.predict(X)
            loss = self._mse_loss(predictions, y)
            print("Epoch = ", str(epoch + 1), " - ", "Loss = ", str(loss))

    def predict(self, X):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))

        preds = []
        for x in X_new:
            pred = self._forward(x)[-1][-1]
            preds.append(pred)

        preds = np.array(preds)
        return preds
