import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module
import torch.nn.functional as F


class DqnAgentNetwork(Module):

    NumberOfNeuronsFirstLayer = 164
    # NumberOfNeuronsSecondLayer = 128
    # NumberOfNeuronsThirdLayer = 64

    def __init__(self, rl_environment):
        super().__init__()

        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.action_dimensions = rl_environment.action_space.n

    def initialize_network(self):

        self.dense_linear_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_linear_layer_2 = nn.Linear(self.NumberOfNeuronsFirstLayer, self.action_dimensions)

    def get_state_action_values(self, state):

        output_of_layer_1 = nn_func.relu(self.dense_linear_layer_1(state))
        output_of_layer_2 = self.dense_linear_layer_2(output_of_layer_1)
        return output_of_layer_2
