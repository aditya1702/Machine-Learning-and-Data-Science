import numpy as np
from utils.variable import Variable


class Graph():
    def __init__(self, mine_maze = None):
        self.mine_maze = mine_maze
        self.graph_maze = np.empty(shape = self.mine_maze.shape, dtype = object)

    def create_graph_from_maze(self):
        for row in range(len(self.mine_maze)):
            for column in range(len(self.mine_maze)):
                self.graph_maze[row, column] = Node(value = self.mine_maze[row, column],
                                                    row = row,
                                                    column = column)

        # Left
        for row in range(len(self.mine_maze)):
            for column in range(len(self.mine_maze)):
                try:
                    if column - 1 >= 0:
                        self.graph_maze[row, column].left = self.graph_maze[row, column - 1]
                except Exception:
                    continue

        # Right
        for row in range(len(self.mine_maze)):
            for column in range(len(self.mine_maze)):
                try:
                    self.graph_maze[row, column].right = self.graph_maze[row, column + 1]
                except Exception:
                    continue

        # Up
        for row in range(len(self.mine_maze)):
            for column in range(len(self.mine_maze)):
                try:
                    if row - 1 >= 0:
                        self.graph_maze[row, column].up = self.graph_maze[row - 1, column]
                except Exception:
                    continue

        # Down
        for row in range(len(self.mine_maze)):
            for column in range(len(self.mine_maze)):
                try:
                    self.graph_maze[row, column].down = self.graph_maze[row + 1, column]
                except Exception:
                    continue
