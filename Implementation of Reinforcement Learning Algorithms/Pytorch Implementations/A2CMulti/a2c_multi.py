# coding=utf-8
import matplotlib.pyplot as plt
import torch
import math
import gym
import numpy as np
from torch.optim import Adam
from torch import LongTensor
from torch.nn.utils import clip_grad_norm

from .actor_critic_network import ActorCriticNetwork
from .utils import Utils
from .envs import Envs


class A2CMultiAgent:

    LearningRate = 3e-3
    MaximumNumberOfEpisodes = 10000
    MaximumNumberOfEpisodeSteps = 20
    Gamma = 0.95
    EntropyBeta = 0.0001
    NumberOfEnvironments = 8

    def __init__(self,
                 rl_environment,
                 actor_critic_network = None,
                 plot_environment_statistics = False):

        # Initialize multiple environments for multiple actors
        self.rl_environments = Envs(rl_environment = rl_environment)

        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.episode_counter = 1
        self.utils = Utils()

        # Initialize global network
        if actor_critic_network is None:
            self.actor_critic_network = ActorCriticNetwork(rl_environment = rl_environment)
            self.actor_critic_network.initialize_network()

        # Initialize optimizer
        self.optimizer = Adam(self.actor_critic_network.parameters(), lr = self.LearningRate)

        # The categorical distribution assigns the respective probabilities to the categories and samples them. For
        # eg. if we have 2 actions - [0,1] with the probabilities being [0.6, 0.4] then 0 is sampled 60% of the times
        #  and 1 is sampled 40% of the times.
        self.distribution = torch.distributions.Categorical

    def train(self, rl_environment = None):
        """
        This function fits a single agent on the environment
        """

        while self.rl_environments.finished_episodes < self.MaximumNumberOfEpisodes:
            self._reset_episode_storage_buffers()

            # Get the initial states from all the environments simultaneously
            states = self.rl_environments._reset()
            for step in range(self.MaximumNumberOfEpisodeSteps):
                actions = self._select_action(states)

                # Take a step. Instead of a single environment step, this steps through all te environments
                # for the multiple actors
                next_states, rewards, dones, infos = self.rl_environments._step(actions)

                # Save the environment transitions to respective local buffers
                self.state_buffer.append(states)
                self.action_buffer.append(actions)
                self.reward_buffer.append(rewards)
                self.done_buffer.append(dones)

                states = next_states

            # We check whether any one of our multiple actors has finished it's episode resulting in Done = True. Only then
            # do we optimize the model. No point in optimizing it when no episode is finished since it will give the same reward
            # signals and no training will occur.
            self.done_list = np.resize(np.array(self.done_buffer), (self.MaximumNumberOfEpisodeSteps * self.NumberOfEnvironments))
            if True in self.done_list:
                self._optimize_model()

        print(self._test_model())

        if self.plot_environment_statistics:
            self.utils._plot_environment_statistics(self.rl_environments.reward_per_episode)

    def _test_model(self):
        score = 0
        rl_environment = gym.make("CartPole-v0")
        state = rl_environment.reset()
        while True:
            state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state))
            action_probabilities, _ = self.actor_critic_network.get_action_probs_and_state_value(state_tensor)
            max_value, max_index = action_probabilities.max(0)
            action = max_index.data[0]
            next_state, reward, done, thing = rl_environment.step(action)
            score += reward
            if done:
                break
            state = next_state
        return score

    def _select_action(self, states):
        """
        The function which gets the discrete actions from the neural network

        :param states (obj:`list`): a list of current states of the environments of multiple actors
        """

        state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(states))
        action_probabilities, _ = self.actor_critic_network.get_action_probs_and_state_value(state_tensor)
        discrete_actions = self.distribution(probs = action_probabilities).sample()
        return discrete_actions

    def _optimize_model(self):
        """
        This function optimizes the actor-critic neural network.
        """

        # The method is exactly similar to the single-agent A2C. The only difference being that we simultaneously optimize
        # across all the actors and environments.
        last_states_of_buffer = self.state_buffer[-1]
        last_dones_of_buffer = self.done_buffer[-1]

        next_state_tensors = self.utils.numpy_array_to_torch_tensor(np.array(last_states_of_buffer))
        _, last_state_returns = self.actor_critic_network.get_action_probs_and_state_value(next_state_tensors)
        last_state_returns = self.utils.torch_tensor_to_numpy_array(last_state_returns)
        next_state_returns = last_state_returns * np.invert(last_dones_of_buffer)

        # Calculate the state return values for all the states across all the environments (actors)
        self.reward_buffer.reverse()
        self.done_buffer.reverse()
        self.target_buffer[0] = next_state_returns
        for index in range(1, len(self.reward_buffer)):
            current_state_returns = self.reward_buffer[index] + self.Gamma * next_state_returns
            current_state_returns = current_state_returns * np.invert(self.done_buffer[index])
            self.target_buffer[index] = current_state_returns
            next_state_returns = current_state_returns
        self.target_buffer = np.flip(self.target_buffer, 0)

        total_loss = self._get_total_loss()
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm(self.actor_critic_network.parameters(), 0.5)
        self.optimizer.step()

    def _get_total_loss(self):
        """
        The function calculates the total loss of the critic and the actor.
        """

        # We convert our episode buffers to pytorch tensors. The big difference here is that we flatten the tensors and optimize across the values
        # of all the actors simultaneously.
        state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(self.state_buffer)).view(self.NumberOfEnvironments * self.MaximumNumberOfEpisodeSteps, 4)
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(self.action_buffer), tensor_type = LongTensor).view(self.NumberOfEnvironments * self.MaximumNumberOfEpisodeSteps, 1)
        target_batch_tensor = self.utils.numpy_array_to_torch_tensor(self.target_buffer).view(self.NumberOfEnvironments * self.MaximumNumberOfEpisodeSteps, 1)

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

        self.state_buffer = list()
        self.action_buffer = list()
        self.reward_buffer = list()
        self.done_buffer = list()
        self.target_buffer = np.zeros((self.MaximumNumberOfEpisodeSteps, self.NumberOfEnvironments))
        self.cumulative_score_buffer = list()