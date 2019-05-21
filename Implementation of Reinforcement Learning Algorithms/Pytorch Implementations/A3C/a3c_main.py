# coding=utf-8
import math
import sys
import random
from collections import namedtuple
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp

from .shared_network import ActorCriticNetwork
from .shared_optimizer import SharedAdam
from .a3c_worker import A3CWorker


class A3CMain:

    LearningRate = 0.0002

    def __init__(self,
                 rl_environment,
                 actor_critic_network = None,
                 plot_environment_statistics = False):
        self.rl_environment = rl_environment
        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.reward_per_episode = dict()

        # Initialize global network
        if actor_critic_network is None:
            self.global_actor_critic_network = ActorCriticNetwork(rl_environment = self.rl_environment)
            self.global_actor_critic_network.initialize_network()
        self.global_actor_critic_network.share_memory()

        # Initialize shared optimizer
        self.shared_optimizer = SharedAdam(self.global_actor_critic_network.parameters(), lr = self.LearningRate)

        # Initialize global counters and variables
        self.global_episode_counter = mp.Value("i", 0)
        self.global_episode_reward = mp.Value("d", 0.)
        self.reward_per_episode_queue = mp.Queue()

        self.number_of_workers = mp.cpu_count() - 1

    def initialize_workers(self):
        """
        This function initializes the worker threads for the A3C algorithm.
        """

        self.workers = [A3CWorker(rl_environment = self.rl_environment,
                                  global_network = self.global_actor_critic_network,
                                  shared_optimizer = self.shared_optimizer,
                                  global_episode_counter = self.global_episode_counter,
                                  global_episode_reward = self.global_episode_reward,
                                  reward_per_episode_queue = self.reward_per_episode_queue,
                                  worker_id = worker_index) for worker_index in range(self.number_of_workers)]

    def train(self, rl_environment = None):
        """
        This function starts the worker threads and at the same time keeps storing the respective rewards
        from the workers.
        """

        for worker in self.workers:
            worker.start()

        if self.plot_environment_statistics:
            self._plot_environment_statistics()

    def _plot_environment_statistics(self):

        reward_index = 1
        while True:
            r = self.reward_per_episode_queue.get()
            reward_index += 1
            if r is not None:
                self.reward_per_episode[reward_index] = r
            else:
                break

        # Join the worker threads
        for worker in self.workers:
            worker.join()

        total_episodes = list(self.reward_per_episode.keys())
        total_rewards = list(self.reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()