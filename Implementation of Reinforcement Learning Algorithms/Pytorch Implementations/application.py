# -*- coding: utf-8 -*-
import gym

from DQN.dqn_agent import DqnAgent
from DDPG.ddpg_agent import Ddpg
from A3C.a3c_main import A3CMain
from A2C.a2c import A2C
from A2CMulti.a2c_multi import A2CMultiAgent
from TRPO.trpo_main import TRPO


class RLApplication():
    """
    RL - Application

    The main class where agents for different RL algorithms are initialized and then they train on
    an environment
    """

    # Agents in the application
    DqnAgentNameString = "dqn"
    DdpgAgentNameString = "ddpg"
    A3CAgentNameString = "a3c"
    A2CAgentNameString = "a2c"
    A2CMultiAgentNameString = "a2c_multi"
    TRPOAgentNameString = "trpo"

    def __init__(self,
                 agent_name = "random_agent",
                 plot_environment_statistics = True):
        self.plot_environment_statistics = plot_environment_statistics
        self.agent_name = agent_name

    def run(self):
        global agent, rl_environment
        if self.agent_name == self.DqnAgentNameString:
            rl_environment = gym.make('CartPole-v0').unwrapped
            agent = DqnAgent(rl_environment = rl_environment,
                             plot_environment_statistics = self.plot_environment_statistics)
        elif self.agent_name == self.DdpgAgentNameString:
            rl_environment = gym.make('Pendulum-v0').env
            agent = Ddpg(rl_environment = rl_environment,
                         plot_environment_statistics = self.plot_environment_statistics)
        elif self.agent_name == self.A3CAgentNameString:
            rl_environment = gym.make('Pendulum-v0').unwrapped
            agent = A3CMain(rl_environment = rl_environment,
                            plot_environment_statistics = self.plot_environment_statistics)
            agent.initialize_workers()
        elif self.agent_name == self.A2CAgentNameString:
            rl_environment = gym.make("CartPole-v0").env
            agent = A2C(rl_environment = rl_environment,
                        plot_environment_statistics = self.plot_environment_statistics)
        elif self.agent_name == self.A2CMultiAgentNameString:
            rl_environment = gym.make("CartPole-v0")
            agent = A2CMultiAgent(rl_environment = rl_environment,
                                  plot_environment_statistics = self.plot_environment_statistics)
        elif self.agent_name == self.TRPOAgentNameString:
            rl_environment = gym.make("Pendulum-v0").unwrapped
            agent = TRPO(rl_environment = rl_environment,
                         plot_environment_statistics = self.plot_environment_statistics)

        agent.train(rl_environment)
        agent.test_agent(rl_environment)

if __name__ == "__main__":
    app = RLApplication(agent_name = "dqn")
    app.run()