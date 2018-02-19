import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn import Module


class ActorCriticNetwork(Module):

    # Initialize neural agent parameters
    NumberOfNeuronsFirstLayer = 64
    NumberOfNeuronsSecondLayer = 128
    NumberOfNeuronsThirdLayer = 64

    def __init__(self, rl_environment):
        super().__init__()

        # Initialize network input and output dimensions
        self.state_dimensions = rl_environment.observation_space.shape[0]
        self.action_dimensions = rl_environment.action_space.n
        self.state_value_dimensions = 1

    def initialize_network(self):

        # common linear layers for both actor and critic
        self.dense_layer_1 = nn.Linear(self.state_dimensions, self.NumberOfNeuronsFirstLayer)
        self.dense_layer_2 = nn.Linear(self.NumberOfNeuronsFirstLayer, self.NumberOfNeuronsSecondLayer)
        self.dense_layer_3 = nn.Linear(self.NumberOfNeuronsSecondLayer, self.NumberOfNeuronsThirdLayer)

        # actor layer outputting log probabilities
        self.actor_layer = nn.Linear(self.NumberOfNeuronsThirdLayer, self.action_dimensions)

        # critic layer outputting state value
        self.critic_layer = nn.Linear(self.NumberOfNeuronsThirdLayer, self.state_value_dimensions)

    def get_action_probs_and_state_value(self, state):

        output_of_dense_layer_1 = nn_func.relu(self.dense_layer_1(state))
        output_of_dense_layer_2 = nn_func.relu(self.dense_layer_2(output_of_dense_layer_1))
        output_of_dense_layer_3 = nn_func.relu(self.dense_layer_3(output_of_dense_layer_2))

        # Get probabilities for actions and state value
        action_probabilities = nn_func.softmax(self.actor_layer(output_of_dense_layer_3))
        state_value = self.critic_layer(output_of_dense_layer_3)
        return action_probabilities, state_value
