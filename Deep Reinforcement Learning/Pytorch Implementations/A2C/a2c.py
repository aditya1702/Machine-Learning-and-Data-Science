# coding=utf-8
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import Adam
from torch import LongTensor

from .actor_critic_network import ActorCriticNetwork
from .utils import Utils


class A2C:

    LearningRate = 3e-3
    MaximumNumberOfEpisodes = 1000
    MaximumNumberOfEpisodeSteps = 20
    Gamma = 0.95
    EntropyBeta = 0.0001

    def __init__(self,
                 rl_environment,
                 actor_critic_network = None,
                 plot_environment_statistics = False):
        self.rl_environment = rl_environment
        self.plot_environment_statistics = plot_environment_statistics
        self.total_reward_gained = 0
        self.reward_per_episode = dict()

        # Initialize actor-critic network
        if actor_critic_network is None:
            self.actor_critic_network = ActorCriticNetwork(rl_environment = self.rl_environment)
            self.actor_critic_network.initialize_network()

        # Initialize optimizer
        self.optimizer = Adam(self.actor_critic_network.parameters(), lr = self.LearningRate)

        self.total_reward_gained = 0
        self.episode_counter = 1
        self.utils = Utils()

        # The categorical distribution assigns the respective probabilities to the categories and samples them. For
        # eg. if we have 2 actions - [0,1] with the probabilities being [0.6, 0.4] then 0 is sampled 60% of the times
        #  and 1 is sampled 40% of the times.
        self.distribution = torch.distributions.Categorical

    def train(self, rl_environment = None):
        """
        This function trains a single agent on the environment
        """

        while self.episode_counter < self.MaximumNumberOfEpisodes:
            self._reset_episode_storage_buffers()
            self.total_reward_gained = 0

            # The data collection phase: Contrary to the DQN, an A2C agent does not have a replay memory. Hence,
            # it plays in the environment for some number of steps and collects its experiences to the different
            # episode storage buffers. This data is then used for optimizing the neural network weights.
            state = self.rl_environment.reset()
            for step in range(self.MaximumNumberOfEpisodeSteps):
                action = self._select_action(state)

                # Take a step
                next_state, reward, done, info = self.rl_environment.step(action)
                self.total_reward_gained += reward

                # Save the environment transitions to respective local buffers
                self.state_buffer.append(state)
                self.action_buffer.append(action)
                self.reward_buffer.append(reward)
                self.done_buffer.append(done)

                state = next_state
                if done:
                    state = self.rl_environment.reset()
                    print(self.total_reward_gained)
                    self.reward_per_episode[self.episode_counter] = self.total_reward_gained
                    self.episode_counter += 1

            # We optimize the model after collecting some amount of training data. This phase can also be called
            # the reflecting phase. The agent reflects on its previous actions to see which ones were good and which
            # ones were bad.
            self._optimize_model()

        if self.plot_environment_statistics:
            self._plot_environment_statistics()

    def _select_action(self, state):
        """
        The function which gets the discrete action from the neural network

        :param state (obj:`float`): the current state of the environment
        """

        state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state))
        action_probabilities, _ = self.actor_critic_network.get_action_probs_and_state_value(state_tensor)

        # Sample the discrete actions [0,1] using their respective probabilities
        discrete_action_tensor = self.distribution(probs = action_probabilities).sample()

        # Get the value of the discrete action tensor
        discrete_action = discrete_action_tensor.data[0]
        return discrete_action

    def _optimize_model(self):
        """
        This function optimizes the actor-critic neural network.
        """

        # We check the last state we stored in our training buffer and correspondingly whether that last state was
        # a terminal one or not indicated by the done variable.
        last_state_of_buffer = self.state_buffer[-1]
        last_done_of_buffer = self.done_buffer[-1]
        next_state_return = 0.

        # If the last stored state is not a terminal state, then its value can be estimated by the critic part of
        # our network. Terminal states have a state return value of 0.
        if not last_done_of_buffer:
            last_state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(last_state_of_buffer))
            _, last_state_return = self.actor_critic_network.get_action_probs_and_state_value(last_state_tensor)
            next_state_return = last_state_return.data[0]

        # Now we start calculating the returns of each state in our training data from reverse.
        self.reward_buffer.reverse()
        self.done_buffer.reverse()

        # Append the return of the last state in the training state buffer
        self.target_buffer.append(next_state_return)
        for index in range(1, len(self.reward_buffer)):
            current_state_return = 0.

            # If the state is a terminal state then its return is 0 else given by the below formula.
            if not self.done_buffer[index]:
                current_state_return = self.reward_buffer[index] + self.Gamma * next_state_return
            self.target_buffer.append(current_state_return)
            next_state_return = current_state_return
        self.target_buffer.reverse()

        self.optimizer.zero_grad()
        total_loss = self._get_total_loss()
        total_loss.backward()
        self.optimizer.step()

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