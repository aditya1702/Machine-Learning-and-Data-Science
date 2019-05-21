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


np.seterr(all = "warn")
class VariationalAutoencoder():

    SigmoidActivation = "sigmoid"
    ReLUActivation = "relu"
    LinearActivation = "linear"
    LeakyReLUActivation = "lrelu"

    def __init__(self,
                 learning_rate = 0.04,
                 batch_size = 32,
                 num_hidden_layers = None,
                 num_neurons_each_layer = None,
                 z_shape = 4,
                 epochs = 10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.z_shape = z_shape

        self.activations_functions = {
            self.SigmoidActivation: self._sigmoid,
            self.LeakyReLUActivation: self._leaky_relu,
            self.ReLUActivation: self._relu,
            self.LinearActivation: self._linear
        }
        self.activations_derivatives = {
            self.SigmoidActivation: self._sigmoid_derivative,
            self.LeakyReLUActivation: self._leaky_relu_derivative,
            self.ReLUActivation: self._relu_derivative,
            self.LinearActivation: self._linear_derivative
        }

        # Activations for Encoder and Decoder
        self.encoder_activations = [self.LeakyReLUActivation] * self.num_hidden_layers + [self.LinearActivation]
        self.decoder_activations = [self.ReLUActivation] * self.num_hidden_layers + [self.SigmoidActivation]

        self.num_neurons_each_encoder_layer = self.num_neurons_each_layer
        self.num_neurons_each_decoder_layer = self.num_neurons_each_layer[::-1]


    def _sigmoid(self, x):
        x = np.select([x < 0, x >= 0], [np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x))])
        return x

    def _relu(self, x):
        return np.maximum(0, x)

    def _leaky_relu(self, x):
        return np.maximum(0, x)

    def _linear(self, x):
        return x

    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu_derivative(self, x):
        return (np.ones_like(x) * (x > 0))

    def _leaky_relu_derivative(self, x):
        return

    def _linear_derivative(self, x):
        return np.ones_like(x)

    def _binary_cross_entropy_loss(self, y_hat, y):
        loss = np.sum(-y * np.log(y_hat + 1e-15) - (1 - y) * np.log(1 - y_hat + 1e-15))
        return loss

    def _kl_divergence(self, mu, log_var):
        return -0.5 * np.sum(1 + log_var - np.power(mu, 2) - np.exp(log_var))

    def _encoder(self, X):
        encoder_out = []

        for curr_layer in self.encoder_layers:
            encoder_out.append([])

            # Get the activation for this layer and its function
            activation_for_this_layer = self.encoder_activations[curr_layer]
            activation_function = self.activations_functions[activation_for_this_layer]

            if curr_layer == 0:
                previous_layer_output = X
            else:
                previous_layer_output = encoder_out[curr_layer - 1].copy()
                previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1, axis = 1)

            if curr_layer != self.encoder_layers[-1]:
                encoder_out[curr_layer] = activation_function(previous_layer_output @ self.encoder_weights[curr_layer].T)
            else:
                encoder_weights_last_layer = np.transpose(self.encoder_weights[curr_layer], axes = (0, 2, 1))
                encoder_out[curr_layer] = activation_function(previous_layer_output @ encoder_weights_last_layer)

        encoder_out = np.array(encoder_out)
        mu, log_var = encoder_out[-1][0], encoder_out[-1][1]
        return mu, log_var, encoder_out

    def _decoder(self, z):

        decoder_out = []

        for curr_layer in self.decoder_layers:
            decoder_out.append([])

            # Get the activation for this layer and its function
            activation_for_this_layer = self.decoder_activations[curr_layer]
            activation_function = self.activations_functions[activation_for_this_layer]

            if curr_layer == 0:
                previous_layer_output = z
            else:
                previous_layer_output = decoder_out[curr_layer - 1].copy()
            previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1, axis = 1)

            decoder_out[curr_layer] = activation_function(previous_layer_output @ self.decoder_weights[curr_layer].T)

        xhat_batch = decoder_out[-1]
        return xhat_batch, decoder_out

    def _forward(self, X):

        # Encode
        mu, log_var, encoder_out = self._encoder(X)

        # Reparametrization trick to sample z from gaussian. First sample x from standard normal distribution.
        # Then we use z = mu + sigma*x to get our latent variable.
        self.rand_sample = np.random.standard_normal(size = (self.batch_size, self.z_shape))
        self.sample_z = mu + np.exp(log_var * .5) * self.rand_sample

        # Decode
        xhat_batch, decoder_out = self._decoder(self.sample_z)

        return mu, log_var, xhat_batch, encoder_out, decoder_out

    def _backward_decoder(self, y, decoder_out):

        decoder_output_derivatives = deepcopy(decoder_out)
        decoder_weight_derivatives = deepcopy(self.decoder_weights)

        # We calculate weight derivatives for each data row in the batch and average the
        # derivatives at the end.
        decoder_weight_derivatives = [decoder_weight_derivatives] * self.batch_size

        # Compute the output derivatives
        layers_reversed = self.decoder_layers[::-1]
        for curr_layer in layers_reversed:
            next_layer = curr_layer + 1

            # For the last layer simply use the formula
            if curr_layer == self.total_decoder_layers - 1:
                decoder_output_derivatives[curr_layer] = -y/(decoder_out[curr_layer] + 1e-16) + \
                                                    (1 - y) * 1/(1 - decoder_out[curr_layer] + 1e-16)
                continue

            # Get the activation derivative function for next layer
            activation_for_next_layer = self.decoder_activations[next_layer]
            activation_derivative = self.activations_derivatives[activation_for_next_layer]

            # The next layer output derivatives
            next_layer_output_derivatives = decoder_output_derivatives[next_layer]

            # Calculate the activation derivative. Add a 1 for the bias weight
            current_layer_output = decoder_out[curr_layer].copy()
            current_layer_output = np.insert(current_layer_output, obj = 0, values = 1, axis = 1)
            next_layer_activation_derivatives = activation_derivative(current_layer_output @ self.decoder_weights[next_layer].T)

            # Remove the bias from the weights. Bias output derivative is 1.
            next_layer_weights_without_bias = self.decoder_weights[next_layer][:, 1:]

            # Cycle through the batch of next layer activation derivatives
            for batch_index, next_layer_activation_derivative in enumerate(next_layer_activation_derivatives):
                next_layer_activation_derivative = next_layer_activation_derivative.reshape(-1, 1)

                # Multiply each neuron's activation derivative with its weights. This is the Hadmard product
                second_term = next_layer_activation_derivative * next_layer_weights_without_bias

                # Sum over all the neurons in the next layer to get the output derivative for each
                # neuron in the current layer. This is because each neuron contributes to all the neurons
                # in the next layer.
                decoder_output_derivatives[curr_layer][batch_index] = next_layer_output_derivatives[batch_index] @ second_term

        # Update the weights using the output derivative calculated above
        for curr_layer in layers_reversed:

            # Get the activation for this layer and its derivative function
            activation_for_this_layer = self.decoder_activations[curr_layer]
            activation_derivative = self.activations_derivatives[activation_for_this_layer]

            # If first layer then use the data as the previous layer.
            if curr_layer == 0:
                previous_layer_output = self.sample_z
            else:
                prev_layer = curr_layer - 1
                previous_layer_output = decoder_out[prev_layer].copy()
            previous_layer_output = np.insert(previous_layer_output, obj = 0, values = 1, axis = 1)

            # Current layer output derivatives
            curr_layer_output_derivatives = decoder_output_derivatives[curr_layer]

            # Get current layer's activation derivatives
            curr_layer_activation_derivatives = activation_derivative(previous_layer_output @ self.decoder_weights[curr_layer].T)
            curr_layer_activation_derivatives = curr_layer_activation_derivatives

            # Cycle through the batch of next layer activation derivatives
            for batch_index, curr_layer_activation_derivative in enumerate(curr_layer_activation_derivatives):
                curr_layer_activation_derivative = curr_layer_activation_derivative.reshape(-1, 1)

                # For the current layer multiply each neuron's activation derivatives with all previous layer outputs.
                curr_layer_weight_derivatives = curr_layer_output_derivatives[batch_index].reshape(-1, 1) * \
                                                curr_layer_activation_derivative * previous_layer_output[batch_index]
                decoder_weight_derivatives[batch_index][curr_layer] = curr_layer_weight_derivatives

        # Average the gradients across batch
        decoder_weight_derivatives = np.mean(decoder_weight_derivatives, axis = 0)

        return decoder_weight_derivatives, decoder_output_derivatives

    def _calculate_mu_derivative(self, decoder_output_derivatives):

        mu_derivatives = np.zeros((self.batch_size, self.z_shape))

        # Add a bias to z
        z_with_bias = np.insert(self.sample_z, obj = 0, values = 1, axis = 1)

        # Activation derivative function for the first layer of decoder
        activation_for_decoder_first_layer = self.decoder_activations[0]
        activation_derivative_func = self.activations_derivatives[activation_for_decoder_first_layer]

        # Activation derivatives for the first layer of decoder.
        decoder_first_layer_activation_derivatives = activation_derivative_func(z_with_bias @ self.decoder_weights[0].T)
        decoder_first_layer_weights_without_bias = self.decoder_weights[0][:, 1:]

        # Cycle through the batch of next layer's activation derivatives
        for batch_index, next_layer_activation_derivative in enumerate(decoder_first_layer_activation_derivatives):
            next_layer_activation_derivative = next_layer_activation_derivative.reshape(-1, 1)
            second_term = next_layer_activation_derivative * decoder_first_layer_weights_without_bias
            mu_derivatives[batch_index] = decoder_output_derivatives[0][batch_index] @ second_term

        return mu_derivatives

    def _calculate_log_var_derivative(self, decoder_output_derivatives, log_var):
        log_var_derivatives = np.zeros((self.batch_size, self.z_shape))

        # Add a bias to z
        z_with_bias = np.insert(self.sample_z, obj = 0, values = 1, axis = 1)

        # Activation derivative function for the first layer of decoder
        activation_for_decoder_first_layer = self.decoder_activations[0]
        activation_derivative_func = self.activations_derivatives[activation_for_decoder_first_layer]

        # Activation derivatives for the first layer of decoder.
        decoder_first_layer_activation_derivatives = activation_derivative_func(z_with_bias @ self.decoder_weights[0].T)
        decoder_first_layer_weights_without_bias = self.decoder_weights[0][:, 1:]

        # Cycle through the batch of next layer's activation derivatives
        for batch_index, next_layer_activation_derivative in enumerate(decoder_first_layer_activation_derivatives):
            next_layer_activation_derivative = next_layer_activation_derivative.reshape(-1, 1)
            second_term = next_layer_activation_derivative * decoder_first_layer_weights_without_bias
            log_var_derivatives[batch_index] = (decoder_output_derivatives[0][batch_index] @ second_term) * \
                                            np.exp(log_var * .5)[batch_index] * 0.5 * self.rand_sample[batch_index]

        return log_var_derivatives

    def _backward_encoder_recon_loss(self, encoder_out, decoder_out, decoder_output_derivatives, log_var):
        encoder_output_derivatives = deepcopy(encoder_out)
        encoder_weight_derivatives = deepcopy(self.encoder_weights)

        # Calculate derivatives of mu and log_var using decoder outputs and derivatives
        mu_derivatives = self._calculate_mu_derivative(decoder_output_derivatives)
        log_var_derivatives = self._calculate_log_var_derivative(decoder_output_derivatives, log_var)

        encoder_output_derivatives_recon = deepcopy(encoder_out)
        encoder_weight_derivatives_recon = deepcopy(self.encoder_weights)

        # We calculate weight derivatives for each data row in the batch and average the
        # derivatives at the end.
        encoder_weight_derivatives_recon = [encoder_weight_derivatives_recon] * self.batch_size
        print(mu_derivatives)
        s
        return

    def _backward_encoder_kl_loss(self):
        return

    def _update_weights(self):
        return

    def _backward(self, xhat_batch, x_batch, encoder_out, decoder_out, log_var):

        # Calculate decoder gradients. We use the reconstruction loss to backpropagate through decoder.
        decoder_weight_derivatives, decoder_output_derivatives = self._backward_decoder(x_batch, decoder_out)

        # Calculate encoder gradients. For encoder, we use both the reconstruction loss and the
        # KL Divergence loss.
        encoder_weight_derivatives_recon_loss = self._backward_encoder_recon_loss(encoder_out,
                                                                                  decoder_out,
                                                                                  decoder_output_derivatives,
                                                                                  log_var)
        encoder_weight_derivatives_kl_loss = self._backward_encoder_kl_loss()

        # Update weights using Adam
        self._update_weights(decoder_weight_derivatives, encoder_weight_derivatives)

        return

    def _initialise_weights(self, input_shape):

        # Encoder Layers
        self.num_neurons_each_encoder_layer.append(2) # 2 for two outputs - mu and sigma
        self.total_encoder_layers = self.num_hidden_layers + 1 # +1 for the last output layer
        self.encoder_layers = range(self.total_encoder_layers)

        # Decoder Layers
        self.num_neurons_each_decoder_layer.append(input_shape) # Last layer of decoder has input shape
        self.total_decoder_layers = self.num_hidden_layers + 1 # +1 for the last output layer
        self.decoder_layers = range(self.total_decoder_layers)

        # Empty weight arrays
        self.encoder_weights = []
        self.decoder_weights = []

        # Initialise encoder weights
        for layer in self.encoder_layers:
            self.encoder_weights.append([])

            number_of_neurons_in_this_layer = self.num_neurons_each_encoder_layer[layer]
            if layer == 0:
                fan_in = input_shape
                previous_layer_shape = fan_in
            else:
                fan_in = self.num_neurons_each_encoder_layer[layer - 1]
                previous_layer_shape =  1 + fan_in

            fan_out = number_of_neurons_in_this_layer
            init_bound = np.sqrt(6. / (fan_in + fan_out))
            if layer != self.encoder_layers[-1]:
                self.encoder_weights[layer] = np.random.uniform(low = -init_bound,
                                                                high = init_bound,
                                                                size = (number_of_neurons_in_this_layer,
                                                                       previous_layer_shape))
            else:
                # Last layer of encoder outputs mu and sigma whose dimensions
                # are of shape z_shape.
                self.encoder_weights[layer] = np.random.uniform(low = -init_bound,
                                                                high = init_bound,
                                                                size = (number_of_neurons_in_this_layer,
                                                                       self.z_shape,
                                                                       previous_layer_shape))



        # Initialise decoder weights
        for layer in self.decoder_layers:
            self.decoder_weights.append([])

            number_of_neurons_in_this_layer = self.num_neurons_each_decoder_layer[layer]
            if layer == 0:
                # Input to decoder is the latent variable constructed from
                # gaussian distribution
                fan_in = self.z_shape
            else:
                fan_in =  self.num_neurons_each_layer[layer - 1]

            fan_out = number_of_neurons_in_this_layer
            previous_layer_shape = 1 + fan_in # +1 for the bias
            init_bound = np.sqrt(6. / (fan_in + fan_out))
            self.decoder_weights[layer] = np.random.uniform(low = -init_bound,
                                                            high = init_bound,
                                                            size = (number_of_neurons_in_this_layer,
                                                                   previous_layer_shape))

        self.encoder_weights = np.array(self.encoder_weights)
        self.decoder_weights = np.array(self.decoder_weights)
        self.old_encoder_weights = deepcopy(self.encoder_weights)
        self.old_decoder_weights = deepcopy(self.decoder_weights)

    def _get_batches(self, X):
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i: i + self.batch_size]

    def fit(self, X):

        # Add a bias column to X
        X_new = np.column_stack((np.ones(len(X)), X))

        # Initialise weights using Glorot Uniform initialiser
        self._initialise_weights(X_new.shape[1])

        # Get batches
        batches = self._get_batches(X_new)

        iterations = 0
        while iterations <= self.epochs:

            # Train using mini-batch SGD
            for x_batch in batches:

                # Forward pass
                mu, log_var, xhat_batch, encoder_out, decoder_out = self._forward(x_batch)

                # Reconstruction Loss - between decoded output and input data
                reconstruction_loss = self._binary_cross_entropy_loss(xhat_batch, x_batch)

                # Calculate KL Divergence between sampled z (Gaussian Distribution: N(mu, sigma))
                # and N(0, 1)
                kl_loss = self._kl_divergence(mu, log_var)

                loss = reconstruction_loss + kl_loss
                loss = loss / self.batch_size

                # Backward pass - for every result in the batch
                # calculate gradient and update the weights using Adam
                self._backward(xhat_batch, x_batch, encoder_out, decoder_out, log_var)
