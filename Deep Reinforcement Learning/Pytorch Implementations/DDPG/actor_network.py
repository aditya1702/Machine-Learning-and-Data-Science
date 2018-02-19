import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class ActorNetwork(Module):

    # Initialize neural agent parameters
    NumberOfNeuronsFirstLayer = 400
    NumberOfNeuronsSecondLayer = 300
    WeightsInitializedRange = 3e-3

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.neural_network_input_dimensions = rl_environment.observation_space.shape[0]
        self.neural_network_output_dimensions = rl_environment.action_space.shape[0]

    def _initialize_weights(self):
        return

    def initialize_network(self):
        """
        This function creates the deep neural-agent with different layers. This neural-agent acts as the state-value
        function approximator for our RL agent. As our agent encounters states in the environment, it will use this
        neural agent to predict the state-action values for all the actions in that particular state.
        """

        # The last value in the output of the last layer will be used for action_amount and the first 4 values will
        # be used for calculating the action_type.
        self.dense_linear_layer_1 = nn.Linear(self.neural_network_input_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_linear_layer_2 = nn.Linear(self.NumberOfNeuronsFirstLayer, self.NumberOfNeuronsSecondLayer)
        self.dense_linear_layer_3 = nn.Linear(self.NumberOfNeuronsSecondLayer, self.neural_network_output_dimensions)

        self._initialize_weights()

    def get_action(self, state):
        """
        This implements a forward pass of the neural-agent. It takes the state as input and produces state-values
        for each action of the environment.

        :param state (obj:`dict`): the current state in which the agent is present.
        :return: the state-action values of the input state
        """

        output_of_dense_layer_1 = nn_func.relu(self.dense_linear_layer_1(state))
        output_of_dense_layer_2 = nn_func.relu(self.dense_linear_layer_2(output_of_dense_layer_1))
        output_of_dense_layer_3 = nn_func.tanh(self.dense_linear_layer_3(output_of_dense_layer_2))
        return output_of_dense_layer_3
