import argparse
import sys
from utils.environment import Environment
from agents.base_agent import BaseAgent
import numpy as np
import matplotlib.pyplot as plt
from agents.prob_csp_agent import ProbCSPAgent
from agents.csp_agent import CSPAgent
from agents.bonus_csp_agent import BonusCSPAgent


class MineSweeper():
    BasicAgent = "base_agent"
    CSPAgent = "csp_agent"

    def __init__(self,
                 ground_dimension = None,
                 mine_density = None,
                 agent_name = None,
                 visual = False,
                 end_game_on_mine_hit = True,
                 bonus_uncertain_p = 0.0):
        self.ground_dimension = ground_dimension
        self.mine_density = mine_density
        self.agent_name = agent_name
        self.visual = visual
        self.end_game_on_mine_hit = end_game_on_mine_hit
        self.bonus_uncertain_p = bonus_uncertain_p
        self.use_probability_agent = False

        if self.bonus_uncertain_p > 0:
            self.use_probability_agent = True

    def create_environment(self):

        # Create the maze
        self.env = Environment(n = self.ground_dimension,
                               mine_density = self.mine_density,
                               visual = self.visual,
                               end_game_on_mine_hit = self.end_game_on_mine_hit)
        self.env.generate_environment()

    def run(self):

        # Use the agent to find mines in our mine-sweeper environment
        if self.agent_name == self.BasicAgent:
            self.mine_sweeper_agent = BaseAgent(env = self.env)
        elif self.agent_name == self.CSPAgent:
            self.mine_sweeper_agent = CSPAgent(env = self.env,
                                               end_game_on_mine_hit = self.end_game_on_mine_hit)
        else:
            self.mine_sweeper_agent = ProbCSPAgent(env = self.env,
                                                   end_game_on_mine_hit = self.end_game_on_mine_hit,
                                                   use_probability_agent = self.use_probability_agent,
                                                   prob = self.bonus_uncertain_p)

        self.mine_sweeper_agent.play()
        metrics = self.mine_sweeper_agent.get_gameplay_metrics()
        # print("Game won = ", str(metrics["game_won"]))
        print("Number of mines hit = ", str(metrics["number_of_mines_hit"]))
        print("Number of mines flagged correctly = ", str(metrics["number_of_mines_flagged_correctly"]))
        print("Number of cells flagged incorrectly = ", str(metrics["number_of_cells_flagged_incorrectly"]))

        self.env.render_env(100)

    def get_performance(self):
        performance_dict_1 = dict()
        performance_dict_2 = dict()
        for mine_density in np.arange(0.01, 0.2, 0.01):
            print(mine_density)
            performance_dict_1[mine_density] = dict()
            performance_dict_2[mine_density] = dict()

            final_scores_1 = list()
            mines_hit_1 = list()
            correct_mines_1 = list()
            incorrect_mines_1 = list()

            final_scores_2 = list()
            mines_hit_2 = list()
            correct_mines_2 = list()
            incorrect_mines_2 = list()
            for _ in range(10):
                env = Environment(n = 15, mine_density = mine_density, end_game_on_mine_hit = False, visual = False)
                env.generate_environment()

                agent1 = CSPAgent(env = env, end_game_on_mine_hit = False)
                agent1.play()

                agent2 = BonusCSPAgent(env = env, end_game_on_mine_hit = False)
                agent2.play()

                # agent = ProbCSPAgent(env = env,
                #                       end_game_on_mine_hit = False,
                #                       use_probability_agent = self.use_probability_agent,
                #                       prob = 0.3)
                # agent.play()

                metrics_1 = agent1.get_gameplay_metrics()
                final_scores_1.append(metrics_1["final_score"])
                mines_hit_1.append(metrics_1["number_of_mines_hit"])

                metrics_2 = agent2.get_gameplay_metrics()
                final_scores_2.append(metrics_2["final_score"])
                mines_hit_2.append(metrics_2["number_of_mines_hit"])

            final_score_1 = np.mean(final_scores_1)
            num_mines_hit_1 = np.mean(mines_hit_1)

            final_score_2 = np.mean(final_scores_2)
            num_mines_hit_2 = np.mean(mines_hit_2)

            performance_dict_1[mine_density]["final_score"] = final_score_1
            performance_dict_1[mine_density]["number_of_mines_hit"] = num_mines_hit_1

            performance_dict_2[mine_density]["final_score"] = final_score_2
            performance_dict_2[mine_density]["number_of_mines_hit"] = num_mines_hit_2

        mine_densities_1 = performance_dict_1.keys()
        final_scores_1 = [performance_dict_1[density]["final_score"] for density in performance_dict_1]
        mines_hit_1 = [performance_dict_1[density]["number_of_mines_hit"] for density in performance_dict_1]

        mine_densities_2 = performance_dict_2.keys()
        final_scores_2 = [performance_dict_2[density]["final_score"] for density in performance_dict_2]
        mines_hit_2 = [performance_dict_2[density]["number_of_mines_hit"] for density in performance_dict_2]

        plt.plot(mine_densities_1, final_scores_1, marker = 'o', label = "Normal CSP Agent")
        plt.plot(mine_densities_2, final_scores_2, marker = 'o', label = "Bonus CSP Agent")
        plt.xlabel("Mine Density")
        plt.ylabel("Average Final Score")
        plt.savefig('avg_final_score_bonus.png')
        plt.legend()
        plt.close()

        plt.plot(mine_densities_1, mines_hit_1, marker = 'o', label = "Normal CSP Agent")
        plt.plot(mine_densities_2, mines_hit_2, marker = 'o', label = "Bonus CSP Agent")
        plt.xlabel("Mine Density")
        plt.ylabel("Average Density of Mines Hit")
        plt.savefig('avg_density_of_mines_hit_bonus.png')
        plt.legend()
        plt.close()

        return performance_dict_1, performance_dict_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'create AI agents to play mine-sweeper')
    parser.add_argument("-n", "--ground_dimension", default = 10)
    parser.add_argument("-a", "--agent_name", default = "csp_agent")
    parser.add_argument("-d", "--mine_density", default = 0.1)
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-e', "--end_game_on_mine_hit", default = False)
    parser.add_argument("-bp", "--bonus_uncertain_p", default = 0)
    args = parser.parse_args(sys.argv[1:])

    mine_sweeper = MineSweeper(ground_dimension = int(args.ground_dimension),
                               mine_density = float(args.mine_density),
                               agent_name = args.agent_name,
                               visual = args.visual,
                               end_game_on_mine_hit = args.end_game_on_mine_hit,
                               bonus_uncertain_p = float(args.bonus_uncertain_p))

    mine_sweeper.create_environment()
    mine_sweeper.run()


