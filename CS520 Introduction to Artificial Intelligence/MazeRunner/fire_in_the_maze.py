import argparse
import sys
from main import MazeRunner


class FireMaze():

    def __init__(self, maze_dimension, probability_of_obstacles, algorithm, visual, heuristic, fire):
        self.maze_dimension = maze_dimension
        self.probability_of_obstacles = probability_of_obstacles
        self.algorithm = algorithm
        self.visual = visual
        self.heuristic = heuristic
        self.fire = fire

    def run(self):
        maze_runner = MazeRunner(maze_dimension = self.maze_dimension,
                                 probability_of_obstacles = self.probability_of_obstacles,
                                 algorithm = self.algorithm,
                                 visual = self.visual,
                                 heuristic = self.heuristic,
                                 fire = self.fire)

        maze_runner.create_environment()
        maze_runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'generate path-finding algorithms to traverse mazes')
    parser.add_argument("-n", "--maze_dimension", default = 10)
    parser.add_argument("-p", "--probability_of_obstacles", default = 0.22)
    parser.add_argument('-algo', "--path_finding_algorithm", default = "dfs")
    parser.add_argument('-v', "--visual", default = False)
    parser.add_argument('-he', "--heuristic", default = "edit")
    parser.add_argument('-f', "--fire", default = False)
    args = parser.parse_args(sys.argv[1:])

    fire_maze = FireMaze(maze_dimension = int(args.maze_dimension),
                             probability_of_obstacles = float(args.probability_of_obstacles),
                             algorithm = args.path_finding_algorithm,
                             visual = bool(args.visual),
                             heuristic = args.heuristic,
                             fire = args.fire)
    fire_maze.run()
