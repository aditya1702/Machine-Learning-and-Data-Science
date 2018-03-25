# coding=utf-8
import matplotlib.pyplot as plt
import torch
import scipy
from scipy import optimize
from collections import namedtuple
import numpy as np
from torch.autograd import Variable

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .utils import Utils
from .replay_memory import ReplayMemory
from .running_stats import ZFilter


class TRPO:

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    LearningRate = 3e-3
    MaximumNumberOfEpisodes = 1000
    MaximumNumberOfEpisodeSteps = 100
    Gamma = 0.95
    EntropyBeta = 0.0001
    BatchSize = 100
    Tau = 0.97
    L2Regularization = 1e-2

    def __init__(self,
                 rl_environment,
                 policy_network = None,
                 value_network = None,
                 plot_environment_statistics = False):
        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.reward_per_episode = dict()

        # Initialize actor-critic network
        if policy_network is None:
            self.policy_network = PolicyNetwork(rl_environment = rl_environment)
            self.policy_network.initialize_network()

        if value_network is None:
            self.value_network = ValueNetwork(rl_environment = rl_environment)
            self.value_network.initialize_network()

        self.running_state = ZFilter((rl_environment.observation_space.shape[0], ), clip_value = 5)
        self.running_reward = ZFilter((1, ), calculate_mean = False, clip_value = 10)

        self.total_reward_gained = 0
        self.episode_counter = 0
        self.number_of_steps = 0
        self.replay_memory = ReplayMemory()
        self.utils = Utils()

    def train(self, rl_environment):
        """
        This function fits a single agent on the environment
        """

        while True:
            self.replay_memory.reset()
            self.total_reward_gained = 0
            self.number_of_steps = 0

            while self.number_of_steps < self.BatchSize:
                state = rl_environment.reset()
                state = self.running_state(state)
                step = 0

                for step in range(self.MaximumNumberOfEpisodeSteps):
                    action = self._select_action(state)

                    # Take a step
                    next_state, reward, done, info = rl_environment.step(action)
                    next_state = self.running_state(next_state)
                    self.total_reward_gained += reward

                    # Save the environment transitions to the replay memory
                    self.replay_memory.append(self.Transition(state, action, next_state, reward, done))

                    state = next_state
                    if done:
                        break
                self.number_of_steps += (step-1)
                self.episode_counter += 1
                self.reward_per_episode[self.episode_counter] = self.total_reward_gained
            self._optimize_model()

    def _select_action(self, state):
        """
        The function which gets the discrete action from the neural network

        :param state (obj:`float`): the current state of the environment
        """

        state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state))
        action_mean, action_std = self.policy_network.get_action_values(state_tensor)
        action = torch.norm(action_mean, action_std.data[0])
        action = action.data.clamp(-1, 1)
        return action

    def _optimize_model(self):
        """
        This function optimizes the actor-critic neural network.
        """

        # Randomly sample the played episode transitions from replay memory.
        batch_transitions = self.replay_memory.sample(self.BatchSize)
        transition_batch = zip(*batch_transitions)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = map(np.array, list(transition_batch))

        # Convert each transition batches to Pytorch tensors
        state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state_batch))
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(action_batch)).unsqueeze(1)
        reward_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(reward_batch))
        done_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(done_batch))
        state_value_batch_tensor = self.value_network.get_state_value(state_batch_tensor)

        returns_batch_tensor = torch.zeros(action_batch_tensor.size(0), 1)
        deltas_batch_tensor = torch.zeros(action_batch_tensor.size(0), 1)
        advantages_batch_tensor = torch.zeros(action_batch_tensor.size(0), 1)

        previous_return = previous_delta = previous_advantage = 0
        for index in reversed(range(reward_batch_tensor.size(0))):
            returns_batch_tensor[index] = reward_batch_tensor.data[index] + self.Gamma * previous_return * done_batch_tensor.data[index]
            deltas_batch_tensor[index] = reward_batch_tensor.data[index] + self.Gamma * previous_delta * done_batch_tensor.data[index] - state_value_batch_tensor.data[index]
            advantages_batch_tensor[index] = deltas_batch_tensor[index] + self.Gamma * self.Tau * done_batch_tensor.data[index] * previous_advantage

            previous_return = returns_batch_tensor[index, 0]
            previous_delta = deltas_batch_tensor[index, 0]
            previous_advantage = advantages_batch_tensor[index, 0]

        target_batch_tensor = Variable(returns_batch_tensor)
        flattened_params = self.utils.get_flattened_params_from_model(self.value_network).double().numpy()

        def _calculate_value_loss(value_parameters):
            flattened_params_tensor = self.utils.numpy_array_to_torch_tensor(np.array(value_parameters))
            self.utils.set_flattened_params_to_model(self.value_network, flattened_params_tensor.data)
            for param in self.value_network.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            state_value_batch_tensor = self.value_network.get_state_value(state_batch_tensor)
            value_loss = (state_value_batch_tensor - target_batch_tensor).pow(2).mean()

            # weight decay
            for param in self.value_network.parameters():
                value_loss += param.pow(2).sum() * self.L2Regularization
            value_loss.backward()
            value_loss_untensored = value_loss.data.double().numpy()[0]
            flattened_params = self.utils.get_flattened_params_from_model(self.value_network).double().numpy()
            return value_loss_untensored, flattened_params

        flattened_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(_calculate_value_loss, flattened_params, maxiter = 25)
        s

    def _plot_environment_statistics(self):

        total_episodes = list(self.reward_per_episode.keys())
        total_rewards = list(self.reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()