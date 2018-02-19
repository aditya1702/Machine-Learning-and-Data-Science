import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class ValueNetwork(Module):

    # Initialize neural agent parameters
    NumberOfNeuronsFirstLayer = 64
    NumberOfNeuronsSecondLayer = 128

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.state_value_dimensions = 1

    def initialize_network(self):

        self.dense_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_layer_2 = nn.Linear(self.NumberOfNeuronsFirstLayer, self.NumberOfNeuronsSecondLayer)
        self.critic_layer = nn.Linear(self.NumberOfNeuronsSecondLayer, self.state_value_dimensions)

    def get_action_probs_and_state_value(self, state):

        output_of_dense_layer_1 = nn_func.tanh(self.dense_layer_1(state))
        output_of_dense_layer_2 = nn_func.tanh(self.dense_layer_2(output_of_dense_layer_1))

        state_value = self.critic_layer(output_of_dense_layer_2)
        return state_value
