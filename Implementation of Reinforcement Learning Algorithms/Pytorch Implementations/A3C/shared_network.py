import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class ActorCriticNetwork(Module):

    # Initialize neural agent parameters
    NumberOfNeuronsFirstLayer = 200
    SigmaValueNoise = 0.001
    InitialWeightsMean = 0.0
    InitialWeightsStandardDeviation = 0.1
    InitialWeightsBias = 0.1

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.action_dimensions = rl_environment.action_space.shape[0]
        self.state_value_dimensions = 1

    def _initialize_weights(self, layers_list):
        for layer in layers_list:
            nn.init.normal(layer.weight, mean = self.InitialWeightsMean, std = self.InitialWeightsStandardDeviation)
            nn.init.constant(layer.bias, self.InitialWeightsBias)

    def initialize_network(self):

        # mu and sigma layers for calculation of continuous action
        self.dense_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.mu = nn.Linear(self.NumberOfNeuronsFirstLayer, self.action_dimensions)
        self.sigma = nn.Linear(self.NumberOfNeuronsFirstLayer, self.action_dimensions)

        # At the same time also output the state-value from the state-value layer
        self.dense_layer_2 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.state_value_layer = nn.Linear(self.NumberOfNeuronsFirstLayer, self.state_value_dimensions)
        self._initialize_weights([self.dense_layer_1, self.mu, self.sigma, self.dense_layer_2, self.state_value_layer])

    def get_action_and_state_value(self, state):

        # Get mu and sigma values
        output_of_dense_layer_1 = nn_func.relu(self.dense_layer_1(state))
        mu = 2 * nn_func.tanh(self.mu(output_of_dense_layer_1))
        sigma = nn_func.softplus(self.sigma(output_of_dense_layer_1)) + self.SigmaValueNoise

        # Get the corresponding state value
        output_of_dense_layer_2 = nn_func.relu(self.dense_layer_2(state))
        state_value = self.state_value_layer(output_of_dense_layer_2)
        return mu, sigma, state_value
