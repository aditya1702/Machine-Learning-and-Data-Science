import argparse
import sys
import numpy as np
import os
from main import MazeRunner


class HardMazeGenerator():

    def __init__(self,
                 maze_dimension = 4,
                 probability_of_obstacles = 0.2,
                 algorithm = 'dfs',
                 metric = "path",
                 heuristic = None,
                 max_iterations = 100,
                 visual = False,
                 fire = False):
        self.maze_dimension = maze_dimension
        self.fire = fire
        self.probability_of_obstacles = probability_of_obstacles
        self.algorithm = algorithm
        self.metric = metric
        self.visual = visual
        self.heuristic = heuristic
        self.max_iterations = max_iterations

        self.image_path = os.curdir + '/output/hard_maze' + "/" + self.algorithm + \
                            "/" + self.metric + "/" + str(self.maze_dimension) + '_' +\
                            str(self.probability_of_obstacles)

    def run(self):

        os.makedirs(self.image_path, exist_ok = True)

        maze_runner = MazeRunner(maze_dimension = self.maze_dimension,
                                 probability_of_obstacles = self.probability_of_obstacles,
                                 algorithm = self.algorithm,
                                 visual = self.visual,
                                 heuristic = self.heuristic,
                                 fire = self.fire)


        self.global_difficult_maze = None
        self.global_difficult_maze_metric = 0
        iteration_count = 0

        while iteration_count < self.max_iterations:
            maze_runner.create_environment()
            maze_runner.run()

            if maze_runner.path_finder.get_final_path_length() == 1 :
                iteration_count = iteration_count + 1
                continue

            filename = self.image_path +os.sep + "master" + "_" + str(iteration_count) + ".png"
            #maze_runner.env.plot_maze(image_path = filename)
            
            current_difficult_maze = maze_runner.env.maze.copy()
            current_difficult_maze_metric = maze_runner.path_finder.get_final_path_length()
            parent_maze = current_difficult_maze.copy()
            while True:

                # Inside Terminate Condition
                for i in range(self.maze_dimension) :
                    for j in range(self.maze_dimension):

                        if (i == 0 and j == 0) or (i == self.maze_dimension - 1
                                                          and j == self.maze_dimension - 1):
                            continue

                        # Store the values of Maximum Difficult metric and the maze
                        maze_runner.env.modify_environment(row = i, column = j)
                        maze_runner.run()

                        if maze_runner.path_finder.get_final_path_length() == 1 :
                            maze_runner.env.reset_environment()
                            continue

                        if self.metric == "path":
                            if maze_runner.path_finder.get_final_path_length() > current_difficult_maze_metric:
                                current_difficult_maze_metric = maze_runner.path_finder.get_final_path_length()
                                current_difficult_maze = maze_runner.env.maze.copy()
                                filename = self.image_path + os.sep + "mater" + "_" + str(iteration_count) +"_"+ str(i) + "_" + str(j)
                                #maze_runner.env.plot_maze(image_path = filename)
                        elif self.metric == "memory":
                            if maze_runner.path_finder.get_maximum_fringe_length() > current_difficult_maze_metric:
                                current_difficult_maze_metric = maze_runner.path_finder.get_maximum_fringe_length()
                                current_difficult_maze = maze_runner.env.maze.copy()
                                filename = self.image_path + os.sep + "master" + "_" + str(iteration_count) +"_"+ str(i) + "_" + str(j)
                                #maze_runner.env.plot_maze(image_path = filename)
                        elif self.metric == "nodes":
                            if maze_runner.path_finder.get_number_of_nodes_expanded() > current_difficult_maze_metric:
                                current_difficult_maze_metric = maze_runner.path_finder.get_number_of_nodes_expanded()
                                current_difficult_maze = maze_runner.env.maze.copy()
                                filename = self.image_path + os.sep + "master" + "_" + str(iteration_count) +"_"+ str(i) + "_" + str(j)
                                #maze_runner.env.plot_maze(image_path = filename)

                        maze_runner.env.reset_environment()

                if np.array_equal(current_difficult_maze, parent_maze):
                    break
                else:
                    parent_maze = current_difficult_maze.copy()
                    maze_runner.env.modify_environment(new_maze = parent_maze.copy())
                    maze_runner.env.set_original_maze(new_maze = parent_maze.copy())

            if self.global_difficult_maze_metric < current_difficult_maze_metric:
                self.global_difficult_maze_metric = current_difficult_maze_metric
                self.global_difficult_maze = current_difficult_maze.copy()
                filename = self.image_path + os.sep + "master" + "_" + "hard" + "_" + str(iteration_count) +"_" +str(self.global_difficult_maze_metric)
                maze_runner.env.plot_maze(image_path = filename)
            # Stopping criteria design
            iteration_count = iteration_count + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate hard mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.2)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-i', "--max_iterations", default = False)
    parser.add_argument('-he', "--heuristic", default = "edit")
    parser.add_argument('-m', "--metric", default = "path")
    args = parser.parse_args(sys.argv[1:])

    hard_maze = HardMazeGenerator(maze_dimension = int(args.maze_dimension),
                                  probability_of_obstacles = float(args.probability_of_obstacles),
                                  algorithm = args.path_finding_algorithm,
                                  visual = bool(args.visual),
                                  max_iterations = int(args.max_iterations),
                                  metric = args.metric,
                                  heuristic = args.heuristic,
                                  fire = False)
    hard_maze.run()
