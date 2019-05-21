import argparse
import sys
from time import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from utils.environment import Environment
from algorithms.path_finding_algorithms import PathFinderAlgorithm


class MazeRunner():

	def __init__(self, maze_dimension, probability_of_obstacles, algorithm, visual, heuristic, fire):
		self.algorithm = algorithm
		self.maze_dimension = maze_dimension
		self.probability_of_obstacles = probability_of_obstacles
		self.visual = visual
		self.heuristic = heuristic
		self.fire = fire

	def create_environment(self, new_maze = None):

		# Create the maze
		self.env = Environment(algorithm = self.algorithm, n = self.maze_dimension, p = self.probability_of_obstacles, fire = self.fire)
		self.env.generate_maze(new_maze = new_maze)

		# Generate graph from the maze
		self.env.create_graph_from_maze()

	def run(self):

		# Run the path finding algorithm on the graph
		self.path_finder = PathFinderAlgorithm(environment = self.env,
											   algorithm = self.algorithm,
											   visual = self.visual,
											   heuristic = self.heuristic)
		self.path_finder.run_path_finder_algorithm()

	def find_solvable_map_size(self):
		dim_list = range(10, 250, 10)
		runtimes = dict()
		for dim in dim_list:
			print("Dim = " + str(dim))

			self.maze_dimension = dim
			self.create_environment()

			start = time()
			self.run()
			end = time() - start
			print(end)
			runtimes[dim] = end
			del self.path_finder

		plt.plot(dim_list, runtimes.values(), marker = "o")
		plt.figure()
		plt.show()

	def plot_performance_metrics(self):
		algo_performance = dict()

		env = Environment(n = 20,
						  p = self.probability_of_obstacles,
						  fire = self.fire)

		# env.generate_maze()
		# env.create_graph_from_maze()

		for n in np.arange(10, 100, 10):
			print(n)

			for algo in ["astar"]:
				for heuristic in ["euclid", "manhattan"]:
					print(algo, heuristic)

					if (algo + "_" + heuristic) not in algo_performance:
						algo_performance[(algo + "_" + heuristic)] = dict()
				
					if algo not in algo_performance:
						algo_performance[algo] = dict()
				
					env.algorithm = algo
					env.create_graph_from_maze()
				
					path_finder = PathFinderAlgorithm(environment = env,
													  algorithm = algo,
													  visual = self.visual,
													  heuristic = self.heuristic)
					path_finder.run_path_finder_algorithm()
					performances = path_finder.performance_dict
				
					if "path_length" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["path_length"] = []

					if "maximum_fringe_size" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["maximum_fringe_size"] = []

					if "number_of_nodes_expanded" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["number_of_nodes_expanded"] = []

					algo_performance[(algo + "_" + heuristic)]['path_length'].append(performances['path_length'])
					algo_performance[(algo + "_" + heuristic)]['maximum_fringe_size'].append(performances['maximum_fringe_size'])
					algo_performance[(algo + "_" + heuristic)]['number_of_nodes_expanded'].append(performances['number_of_nodes_expanded'])

			for algo in ["astar", "thin_astar"]:
				for heuristic in ["euclid", "manhattan"]:
					print(algo, heuristic)

					if (algo + "_" + heuristic) not in algo_performance:
						algo_performance[(algo + "_" + heuristic)] = dict()

					env.algorithm = algo

					path_finder = PathFinderAlgorithm(environment = env,
													  algorithm = algo,
													  q = q,
													  visual = self.visual,
													  heuristic = heuristic)
					path_finder.run_path_finder_algorithm()
					performances = path_finder.performance_dict

					if "path_length" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["path_length"] = []

					if "maximum_fringe_size" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["maximum_fringe_size"] = []

					if "number_of_nodes_expanded" not in algo_performance[(algo + "_" + heuristic)]:
						algo_performance[(algo + "_" + heuristic)]["number_of_nodes_expanded"] = []

					algo_performance[(algo + "_" + heuristic)]['path_length'].append(performances['path_length'])
					algo_performance[(algo + "_" + heuristic)]['maximum_fringe_size'].append(performances['maximum_fringe_size'])
					algo_performance[(algo + "_" + heuristic)]['number_of_nodes_expanded'].append(performances['number_of_nodes_expanded'])

		return algo_performance

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'generate path-finding algorithms to traverse mazes')
	parser.add_argument("-n", "--maze_dimension", default = 10)
	parser.add_argument("-p", "--probability_of_obstacles", default = 0.2)
	parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
	parser.add_argument('-v', "--visual", default = False)
	parser.add_argument('-he', "--heuristic", default = "euclid")
	parser.add_argument('-f', "--fire", default = False)
	args = parser.parse_args(sys.argv[1:])

	maze_runner = MazeRunner(maze_dimension = int(args.maze_dimension),
							 probability_of_obstacles = float(args.probability_of_obstacles),
							 algorithm = args.path_finding_algorithm,
							 visual = bool(args.visual),
							 heuristic = args.heuristic,
							 fire = args.fire)

	maze_runner.create_environment()
	maze_runner.run()