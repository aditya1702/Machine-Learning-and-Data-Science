# coding=utf-8
import sys
from collections import namedtuple
import logging

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.optim as optimizers
from torch.autograd import Variable
from stochastic.diffusion import OrnsteinUhlenbeckProcess

from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .replay_memory import ReplayMemory
from .utils import Utils


class Ddpg:

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    # Initialize learning rate and optimizer parameters
    Gamma = 0.99
    BatchSize = 64
    Tau = 0.001
    ActorLearningRate = 1e-4
    CriticLearningRate = 1e-3

    ReplayMemorySize = 300
    DefaultNumberOfEpisodes = 300
    DefaultNumberOfDaysToPrint = 1

    logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)
    logger = logging.getLogger()

    def __init__(self,
                 rl_environment,
                 actor = None,
                 critic = None,
                 number_of_episodes = DefaultNumberOfEpisodes,
                 plot_environment_statistics = False):
        self.number_of_steps_taken = 0
        self.number_of_parameter_updates = 0
        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.number_of_episodes = number_of_episodes

        self.reward_per_episode = dict()

        # Initialize actor's and target actor's networks
        self.actor = actor
        if self.actor is None:
            self.actor = ActorNetwork(rl_environment)
            self.actor.initialize_network()

            self.actor_target = ActorNetwork(rl_environment)
            self.actor_target.initialize_network()

        # Initialize critic's and target critic's networks
        self.critic = critic
        if self.critic is None:
            self.critic = CriticNetwork(rl_environment)
            self.critic.initialize_network()

            self.critic_target = CriticNetwork(rl_environment)
            self.critic_target.initialize_network()

        # Initialize optimizers
        self.actor_optimizer = optimizers.Adam(self.actor.parameters(), lr = self.ActorLearningRate)
        self.critic_optimizer = optimizers.Adam(self.critic.parameters(), lr=self.CriticLearningRate)

        # Initialize Replay Memory
        self.replay_memory = ReplayMemory()
        self.exploration_process = OrnsteinUhlenbeckProcess(speed = 0.15, vol = 0.2)

        self.loss_function = nn.MSELoss()
        self.utils = Utils()

    def train(self, rl_environment):
        """
        This method trains the agent on the environment

        :param rl_environment (obj:`Environment`): the environment to train the agent on
        """

        for episode_number in range(self.number_of_episodes):
            self.total_reward_gained = 0
            self.current_episode_number = episode_number

            state = rl_environment.reset()
            while True:
                action = self._select_action(state)
                next_state, reward, done, info = rl_environment.step(action)

                self.total_reward_gained += reward
                print(self.total_reward_gained)

                self.replay_memory.append(self.Transition(state, action, next_state, reward, done))

                self._optimize_model()

                if done:
                    self._save_reward_info(reward=self.total_reward_gained)
                    break
                state = next_state
            print("Episode - " + str(self.current_episode_number) + "    " + "Reward - " + str(self.total_reward_gained))

        if self.plot_environment_statistics:
            self._plot_environment_statistics()

    def _select_action(self, state):
        state_pytorch_variable = self.utils.numpy_array_to_torch_tensor(np.array(state))
        state_pytorch_variable.volatile = True
        action = self.actor.get_action(state_pytorch_variable)
        action += self.exploration_process.sample(n = 1)[1]
        action_clamped = action.data.clamp(-1, 1)
        return action_clamped

    def _optimize_model(self):

        if self.replay_memory.get_size() < self.ReplayMemorySize:
            return

        # Sample batch transitions from the replay memory
        batch_transitions = self.replay_memory.sample(self.BatchSize)
        transition_batch = zip(*batch_transitions)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = map(np.array, list(transition_batch))

        # Convert each transition batches to Pytorch tensors
        state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state_batch))
        next_state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(next_state_batch), is_volatile = True)
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(action_batch))
        reward_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(reward_batch))
        not_done_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(1 - done_batch))
        reward_batch_tensor_unsqueezed = reward_batch_tensor.unsqueeze(1)
        not_done_batch_tensor_unsqueezed = not_done_batch_tensor.unsqueeze(1)

        # Calculate TD target for updating critic
        next_action_batch_tensor = self.actor_target.get_action(next_state_batch_tensor)
        next_state_action_values = self.critic_target.get_state_action_values(next_state_batch_tensor, next_action_batch_tensor)
        target_state_action_values = reward_batch_tensor_unsqueezed + (self.Gamma * next_state_action_values * not_done_batch_tensor_unsqueezed)

        # Update critic parameters
        self.critic_optimizer.zero_grad()
        current_state_action_values = self.critic.get_state_action_values(state_batch_tensor, action_batch_tensor.unsqueeze(1))
        value_loss = self.loss_function(current_state_action_values, target_state_action_values)
        value_loss.backward()
        self.critic_optimizer.step()

        # Update actor parameters
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic.get_state_action_values(state_batch_tensor, self.actor.get_action(state_batch_tensor))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Do a soft update of actor and critic target parameters
        self._update_target_parameters(self.actor, self.actor_target)
        self._update_target_parameters(self.critic, self.critic_target)

    def _update_target_parameters(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.Tau) + source_param.data * self.Tau)

    def _save_reward_info(self, reward):
        self.reward_per_episode[self.current_episode_number] = reward

    def test_agent(self, rl_environment):

        state = rl_environment.reset()
        while True:
            action = self._select_action(state)
            rl_environment.render()
            next_state, reward, done, info = rl_environment.step(action)

            reward = max(-1.0, min(reward, 1.0))
            self.total_reward_gained += reward

            if done:
                self._save_reward_info(reward = self.total_reward_gained)
                break
            state = next_state
        print(self.total_reward_gained)

    def _plot_environment_statistics(self):
        total_episodes = list(self.reward_per_episode.keys())
        total_rewards = list(self.reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()