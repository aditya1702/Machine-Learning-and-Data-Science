import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class CriticNetwork(Module):

    # Initialize neural network layer parameters
    NumberOfNeuronsFirstLayer = 400
    NumberOfNeuronsSecondLayer = 300
    WeightsInitializedRange = 3e-3

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.action_dimensions = rl_environment.action_space.shape[0]

    def _initialize_weights(self):
        return

    def initialize_network(self):
        self.dense_linear_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_linear_layer_2 = nn.Linear((self.NumberOfNeuronsFirstLayer + self.action_dimensions), self.NumberOfNeuronsSecondLayer)
        self.dense_linear_layer_3 = nn.Linear(self.NumberOfNeuronsSecondLayer, 1)

        self._initialize_weights()

    def get_state_action_values(self, state, action):
        output_of_dense_linear_layer_1 = nn_func.relu(self.dense_linear_layer_1(state))
        concatenated_output_of_first_layer_and_actions = torch.cat((output_of_dense_linear_layer_1, action), 1)
        output_of_dense_linear_layer_2 = nn_func.relu(self.dense_linear_layer_2(concatenated_output_of_first_layer_and_actions))
        output_of_dense_linear_layer_3 = self.dense_linear_layer_3(output_of_dense_linear_layer_2)
        return output_of_dense_linear_layer_3
