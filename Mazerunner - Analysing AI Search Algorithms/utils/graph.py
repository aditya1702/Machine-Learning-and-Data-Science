import numpy as np
from utils.node import Node


class Graph():
    def __init__(self, maze = None, algorithm = None):
        self.maze = maze
        self.algorithm = algorithm
        self.graph_maze = np.empty(shape = self.maze.shape, dtype = object)

    def create_graph_from_maze(self):
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                if self.maze[row, column] == 0:
                    continue
                self.graph_maze[row, column] = Node(value = self.maze[row, column],
                                                    row = row,
                                                    column = column,
                                                    algorithm = self.algorithm)

        # Left
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    if column - 1 >= 0:
                        self.graph_maze[row, column].left = self.graph_maze[row, column - 1]
                except Exception:
                    continue

        # Right
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    self.graph_maze[row, column].right = self.graph_maze[row, column + 1]
                except Exception:
                    continue

        # Up
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    if row - 1 >= 0:
                        self.graph_maze[row, column].up = self.graph_maze[row - 1, column]
                except Exception:
                    continue

        # Down
        for row in range(len(self.maze)):
            for column in range(len(self.maze)):
                try:
                    self.graph_maze[row, column].down = self.graph_maze[row + 1, column]
                except Exception:
                    continue