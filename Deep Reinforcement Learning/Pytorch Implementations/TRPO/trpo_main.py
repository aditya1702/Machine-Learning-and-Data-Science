# coding=utf-8
import matplotlib.pyplot as plt
import torch
import math
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from torch import LongTensor

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .utils import Utils
from .replay_memory import ReplayMemory
from .running_stats import ZFilter


class TRPO:

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    LearningRate = 3e-3
    MaximumNumberOfEpisodes = 1000
    MaximumNumberOfEpisodeSteps = 10000
    Gamma = 0.95
    EntropyBeta = 0.0001
    BatchSize = 10000

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

        # Our batch_transitions are currently a list of Transition objects. Using zip(*) we convert it to a list
        # containing tuples of all states, actions, next_states, rewards and done batches. It is of the form
        # [(state_batch), (action_batch), (next_state_batch), (reward_batch), (done_batch)]
        transition_batch = zip(*batch_transitions)

        # from_numpy() function of Pytorch takes a numpy array and so we convert the tuples to arrays using map()
        # function.
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = map(np.array, list(transition_batch))
        print(state_batch)
        s

    def _get_total_loss(self):
        """
        The function calculates the total loss of the critic and the actor.
        """

        # We convert our episode buffers to pytorch tensors
        state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.state_buffer))
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.action_buffer), tensor_type = LongTensor)
        target_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.target_buffer))

        action_probabilities_batch, current_state_value_batch = self.actor_critic_network.get_action_probs_and_state_value(state_batch_tensor)
        action_log_probabilities_batch = action_probabilities_batch.log()
        action_log_probabilities_based_on_previous_actions = action_log_probabilities_batch.gather(1, action_batch_tensor)

        # Calculate the advantage from target value and the current state value given by the neural net
        td_advantage_batch = target_batch_tensor - current_state_value_batch

        # Calculate the entropies
        entropy_batch = (action_probabilities_batch * action_log_probabilities_batch).sum(1).mean()

        # Calculate the total loss - critic loss + action loss
        value_loss = td_advantage_batch.pow(2).mean()
        action_loss = -(action_log_probabilities_based_on_previous_actions * td_advantage_batch + self.EntropyBeta * entropy_batch).mean()
        total_loss = value_loss + action_loss
        return total_loss

    def _reset_episode_storage_buffers(self):
        """
        The function resets the worker level storage buffers initialized earlier
        """

        self.state_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, self.target_buffer = list(), list(), list(), list(), list()

    def _plot_environment_statistics(self):

        total_episodes = list(self.reward_per_episode.keys())
        total_rewards = list(self.reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()