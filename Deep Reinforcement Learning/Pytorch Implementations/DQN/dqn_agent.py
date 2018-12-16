# coding=utf-8
import math
import random
from collections import namedtuple

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import display
# from JSAnimation.IPython_display import display_animation

import numpy as np
import torch
import torch.nn.functional as nn_func
import torch.optim as optimizers
from time import sleep

from .dqn_agent_network import DqnAgentNetwork
from .replay_memory import ReplayMemory
from .utils import Utils
# from moviepy.editor import ImageSequenceClip


class DqnAgent:

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    # Initialize learning rate and optimizer parameters
    Gamma = 0.75
    TargetUpdateFreq = 10
    LearningRate = 0.001
    EpsStart = 0.9
    EpsEnd = 0.05
    EpsDecay = 200
    BatchSize = 64

    ReplayMemorySize = 10000
    DefaultNumberOfEpisodes = 2000
    DefaultNumberOfDaysToPrint = 1

    def __init__(self,
                 rl_environment,
                 model = None,
                 target_agent = None,
                 number_of_episodes = DefaultNumberOfEpisodes,
                 plot_environment_statistics = False):
        self.number_of_steps_taken = 0
        self.number_of_parameter_updates = 0
        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.number_of_episodes = number_of_episodes
        self.number_of_actions = rl_environment.action_space.n

        self.reward_per_episode = dict()

        # Initialize primary agent's neural network
        self.model = model
        if self.model is None:
            self.model = DqnAgentNetwork(rl_environment)
            self.model.initialize_network()

        # Initialize target agent's neural network
        self.target_agent = target_agent
        if self.target_agent is None:
            self.target_agent = DqnAgentNetwork(rl_environment)
            self.target_agent.initialize_network()
        self.target_agent.load_state_dict(self.model.state_dict())
        self.target_agent.eval()

        # Initialize optimizer
        self.optimizer = optimizers.Adam(self.model.parameters(), self.LearningRate)

        # Initialize Replay Memory
        self.replay_memory = ReplayMemory()

        self.utils = Utils()

    def train(self, rl_environment):
        """
        This method trains the agent on the game environment

        :param rl_environment (obj:`Environment`): the environment to train the agent on
        """

        for episode_number in range(self.number_of_episodes):
            self.total_reward_gained = 0
            self.current_episode_number = episode_number
            episode_steps = 0

            state = rl_environment.reset()
            while True:
                action = self._select_action(state)
                next_state, reward, done, _ = rl_environment.step(action)

                # negative reward when attempt ends
                if done:
                    if episode_steps < 30:
                        reward -= 10
                    else:
                        reward = -1
                if episode_steps > 100 or episode_steps > 200 or episode_steps > 300:
                    reward += 1
                self.total_reward_gained += reward

                self.replay_memory.append(self.Transition(state, action, next_state, reward, done))

                self._optimize_model()

                if done:
                    self._save_reward_info(reward = self.total_reward_gained)
                    break
                state = next_state
                episode_steps += 1
            print("Episode - " + str(episode_number+1) + "    " + "Reward - " + str(self.total_reward_gained))

        if self.plot_environment_statistics:
            self._plot_environment_statistics()

    def _select_action(self, state):
        random_number = random.random()
        eps_threshold = self.EpsEnd + (self.EpsStart - self.EpsEnd) * math.exp(-1. * self.number_of_steps_taken / self.EpsDecay)
        self.number_of_steps_taken += 1

        if random_number > eps_threshold:
            with torch.no_grad():
                state_pytorch_variable = self.utils.numpy_array_to_torch_tensor(np.array(state)).unsqueeze(0)
            state_action_values = self.model.get_state_action_values(state_pytorch_variable)
            state_action_values = state_action_values.data[0].numpy()
            discrete_greedy_action = np.argmax(state_action_values)
            return discrete_greedy_action
        discrete_random_action = random.randrange(self.number_of_actions)
        return discrete_random_action

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
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(action_batch)).unsqueeze(1)
        action_batch_tensor = action_batch_tensor.long()
        reward_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(reward_batch))
        not_done_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.array(1 - done_batch))

        # Calculate current state action values based on previously chosen actions
        current_state_action_values = self.model.get_state_action_values(state_batch_tensor)
        current_state_values_based_on_selected_actions = current_state_action_values.gather(1, action_batch_tensor)

        # Calculate the next-state values for the agent to converge to. For the target values, we always select the greedy action.
        with torch.no_grad():
            next_state_action_values = self.target_agent.get_state_action_values(next_state_batch_tensor)
        next_state_values_based_on_greedy_action = not_done_batch_tensor * next_state_action_values.max(1)[0]

        # Calculate the td-target
        target_state_values = (next_state_values_based_on_greedy_action * self.Gamma) + reward_batch_tensor
        target_state_values_reshaped = target_state_values.unsqueeze(1)

        loss = nn_func.smooth_l1_loss(current_state_values_based_on_selected_actions, target_state_values_reshaped)

        # Take a step in the direction where the loss is minimized
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network params every few steps
        self.number_of_parameter_updates += 1
        if self.number_of_parameter_updates % self.TargetUpdateFreq == 0:
            self.target_agent.load_state_dict(self.model.state_dict())

    def _save_reward_info(self, reward):
        self.reward_per_episode[self.current_episode_number] = reward

    def test_agent(self, rl_environment):

        state = rl_environment.reset()
        image_frames = []
        episode_steps = 0
        while True:
            sleep(0.1)
            frame = rl_environment.render(mode = 'rgb_array')
            image_frames.append(frame)

            action = self._select_action(state)
            next_state, reward, done, info = rl_environment.step(action)

            if done:
                self._save_reward_info(reward = self.total_reward_gained)
                break
            state = next_state

        self._display_frames_as_gif(image_frames, filename_gif = 'dqn.gif')

    def _display_frames_as_gif(self, frames, filename_gif = None):
        """
        Displays a list of frames as a gif, with controls
        """

        plt.figure(figsize = (frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 50)
        if filename_gif: anim.save('/Users/adityavyas/Desktop/' + filename_gif, writer = 'imagemagick', fps = 20)

    def _plot_environment_statistics(self):
        total_episodes = list(self.reward_per_episode.keys())
        total_rewards = list(self.reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()