import numpy as np


class Envs:

    NumberOfEnvironments = 8

    def __init__(self, rl_environment):
        self.rl_environments_list = self.NumberOfEnvironments * [rl_environment]
        self.cumulative_scores = self.NumberOfEnvironments * [0]
        self.finished_episodes = 0
        self.reward_per_episode = dict()

    def _step(self, actions):
        """
        This function steps through all the environments of the multiple actors

        :param actions (obj:`list`): a list of actions taken by the multiple actors
        """

        self._reset_episode_buffers()
        for env_index in range(self.NumberOfEnvironments):

            rl_environment = self.rl_environments_list[env_index]
            discrete_action_tensor = actions[env_index]
            discrete_action_tensor_value = discrete_action_tensor.data[0]
            next_state, reward, done, info = rl_environment.step(discrete_action_tensor_value)

            self.next_states.append(next_state)
            self.rewards.append(reward)
            self.cumulative_scores[env_index] += reward
            self.dones.append(done)
            self.infos.append(info)

            if done:
                rl_environment.reset()
                self.finished_episodes += 1
                total_reward_gained = np.sum(self.cumulative_scores)
                print(total_reward_gained)
                self.reward_per_episode[self.finished_episodes] = total_reward_gained
                self.cumulative_scores[env_index] = 0

        return self.next_states, self.rewards, self.dones, self.infos

    def _reset_episode_buffers(self):
        self.next_states = list()
        self.rewards = list()
        self.dones = list()
        self.infos = list()

    def _reset(self):
        """
        This function resets all the environments of the actors involved
        """

        reset_states = list()
        for env_index in range(self.NumberOfEnvironments):
            rl_environment = self.rl_environments_list[env_index]
            state = rl_environment.reset()
            reset_states.append(state)
        return reset_states

