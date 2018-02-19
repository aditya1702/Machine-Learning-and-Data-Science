# coding=utf-8
import math
import multiprocessing as mp
import torch
import numpy as np
from .shared_network import ActorCriticNetwork
from .utils import Utils


class A3CWorker(mp.Process):

    WorkerNameString = "Worker"
    MaximumNumberOfEpisodes = 3000
    MaximumNumberOfEpisodeSteps = 200
    UpdateGlobalNetworkParamsStep = 5
    Gamma = 0.9
    EntropyBeta = 0.005

    def __init__(self,
                 rl_environment,
                 global_network,
                 shared_optimizer,
                 global_episode_counter,
                 global_episode_reward,
                 reward_per_episode_queue,
                 worker_id):
        super().__init__()
        self.rl_environment = rl_environment
        self.global_network = global_network
        self.shared_optimizer = shared_optimizer
        self.global_episode_counter = global_episode_counter
        self.global_episode_reward = global_episode_reward
        self.reward_per_episode_queue = reward_per_episode_queue
        self.worker_id = self.WorkerNameString + str(worker_id)

        # Initialize local network
        self.local_actor_critic_network = ActorCriticNetwork(rl_environment = self.rl_environment)
        self.local_actor_critic_network.initialize_network()

        self.total_number_of_steps = 1
        self.utils = Utils()
        self.distribution = torch.distributions.Normal

    def run(self):
        """
        The main function where the worker thread plays in the environment
        """

        while self.global_episode_counter.value < self.MaximumNumberOfEpisodes:
            self._reset_episode_storage_buffers()
            local_episode_reward = 0.
            state = self.rl_environment.reset()
            for step in range(self.MaximumNumberOfEpisodeSteps):
                action = self._select_action(state)
                action_clipped = action.clip(-2, 2)

                # Take a step
                next_state, reward, done, info = self.rl_environment.step(action_clipped)
                local_episode_reward += reward

                # Save the environment transitions to respective local buffers
                self.state_buffer.append(state)
                self.action_buffer.append(action)
                self.reward_buffer.append(reward)

                # The episode ends if either the step reaches the maximum number of episode steps or the agent
                # reaches a terminal state.
                if step == self.MaximumNumberOfEpisodeSteps - 1:
                    done = True
                if self.total_number_of_steps % self.UpdateGlobalNetworkParamsStep == 0 or done:
                    self._update_network_parameters(next_state = next_state, done = done)
                    self._reset_episode_storage_buffers()

                    if done:
                        self.utils.save_environment_info(global_episode_counter = self.global_episode_counter,
                                                          global_episode_reward = self.global_episode_reward,
                                                          local_episode_reward = local_episode_reward,
                                                          worker_id = self.worker_id,
                                                          reward_per_episode_queue = self.reward_per_episode_queue)
                        break
                state = next_state
                self.total_number_of_steps += 1
        self.reward_per_episode_queue.put(None)

    def _select_action(self, state):
        """
        The main function where the worker thread plays in the environment

        :param state (obj:`float`): the current state of the environment
        """

        state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(state))
        self.local_actor_critic_network.training = False
        mu, sigma, state_value = self.local_actor_critic_network.get_action_and_state_value(state_tensor)
        normal_distribution = self.distribution(mean = mu.view(1, ).data, std = sigma.view(1, ).data)
        action_value = normal_distribution.sample().numpy()
        return action_value

    def _update_network_parameters(self, next_state, done):
        """
        This function optimizes the local and global neural networks.

        :param next_state (obj:`float`): the state to which the agent goes after taking an action
        :param done (obj:`Boolean`): this indicates whether the episode ends or not
        """

        next_state_value = 0.
        if not done:
            next_state_tensor = self.utils.numpy_array_to_torch_tensor(np.array(next_state))
            _, _, next_state_value = self.local_actor_critic_network.get_action_and_state_value(next_state_tensor)
            next_state_value = next_state_value.data[0]

        target_value = next_state_value
        self._reverse_episode_storage_buffers()
        for reward in self.reward_buffer:
            target_value = reward + self.Gamma * target_value
            self.target_buffer.append(target_value)

        self.shared_optimizer.zero_grad()
        total_loss = self._get_total_loss()
        total_loss.backward()
        self._ensure_shared_gradients()
        self.shared_optimizer.step()

        self.local_actor_critic_network.load_state_dict(self.global_network.state_dict())

    def _get_total_loss(self):
        """
        The function calculates the total loss of the critic and the actor.
        """

        # We convert our episode buffers to pytorch tensors
        state_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.state_buffer))
        action_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.action_buffer))
        target_batch_tensor = self.utils.numpy_array_to_torch_tensor(np.vstack(self.target_buffer))

        self.local_actor_critic_network.train()
        mu_batch, sigma_batch, current_state_value_batch = self.local_actor_critic_network.get_action_and_state_value(state_batch_tensor)

        # Calculate the advantage from target value and the current state value given by the neural net
        td_advantage_batch = target_batch_tensor - current_state_value_batch

        # Get the log probabilities of the actions
        normal_distribution_batch = self.distribution(mean = mu_batch, std = sigma_batch)
        log_probability_of_action_batch = normal_distribution_batch.log_prob(action_batch_tensor)

        # Calculate the entropies
        entropy_batch = 0.5 * (math.log(2 * math.pi) + torch.log(normal_distribution_batch.std) + 1)

        # Calculate the total loss - critic loss + action loss
        value_loss = td_advantage_batch.pow(2)
        action_loss = -(log_probability_of_action_batch * td_advantage_batch + self.EntropyBeta * entropy_batch)
        total_loss = (value_loss + action_loss).mean()
        return total_loss

    def _reset_episode_storage_buffers(self):
        """
        The function resets the worker level storage buffers initialized earlier
        """

        self.state_buffer, self.action_buffer, self.reward_buffer, self.target_buffer = list(), list(), list(), list()

    def _reverse_episode_storage_buffers(self):
        """
        this function reverses the episode storage buffers
        """

        self.state_buffer.reverse()
        self.action_buffer.reverse()
        self.reward_buffer.reverse()

    def _ensure_shared_gradients(self):

        for shared_param, global_param in zip(self.local_actor_critic_network.parameters(), self.global_network.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = shared_param.grad


