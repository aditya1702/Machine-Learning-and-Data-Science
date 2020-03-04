import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class PolicyNetwork(Module):

    # Initialize neural agent parameters
    NumberOfNeuronsFirstLayer = 64
    NumberOfNeuronsSecondLayer = 128
    NumberOfNeuronsThirdLayer = 64

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.action_dimensions = rl_environment.action_space.shape[0]

    def initialize_network(self):

        self.dense_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_layer_2 = nn.Linear(self.NumberOfNeuronsFirstLayer, self.NumberOfNeuronsSecondLayer)
        self.dense_layer_3 = nn.Linear(self.NumberOfNeuronsSecondLayer, self.NumberOfNeuronsThirdLayer)

        self.action_mean_layer = nn.Linear(self.NumberOfNeuronsThirdLayer, self.action_dimensions)
        self.action_std_layer = nn.Linear(self.NumberOfNeuronsSecondLayer, self.action_dimensions)

    def get_action_values(self, state):

        output_of_dense_layer_1 = nn_func.tanh(self.dense_layer_1(state))
        output_of_dense_layer_2 = nn_func.tanh(self.dense_layer_2(output_of_dense_layer_1))
        output_of_dense_layer_3 = nn_func.tanh(self.dense_layer_3(output_of_dense_layer_2))

        action_mean = self.action_mean_layer(output_of_dense_layer_3)
        action_std = self.action_std_layer(output_of_dense_layer_2)
        return action_mean, action_std
