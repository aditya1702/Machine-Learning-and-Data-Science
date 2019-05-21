import numpy as np


class BaseAgent():

    def __init__(self, env = None):
        self.env = env

    def play(self):
        self.env.click_square(0, 0)
        self.env.render_env()

        old_ground = None
        current_ground = self.env.mine_ground_copy
        while not np.array_equal(old_ground, current_ground):
            old_ground = current_ground.copy()
            self._basic_solver(current_ground)
            current_ground = self.env.mine_ground_copy

    def _basic_solver(self, ground):
        for row in range(ground.shape[0]):
            for column in range(ground.shape[1]):
                if np.isnan(ground[row, column]) or self.env.flags[row, column]:
                    continue
                else:
                    if ground[row, column] == 0:
                        self._query_all_neighbours(row, column)

                    elif ground[row,column] == 8:
                        self._flag_all_neighbours(row, column)

                    else:
                        if self._get_bomb(row, column) == ground[row, column]:
                            self._query_all_neighbours(row, column)
                        elif self._get_unexplored(row, column) == ground[row, column]:
                            self._flag_all_neighbours(row, column)

    def _query_all_neighbours(self, row, column):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (row + i >= 0 and column + j >= 0 and row + i < self.env.mine_ground_copy.shape[0]
                        and column + j < self.env.mine_ground_copy.shape[1] and (not self.env.flags[row + i, column + j])
                        and np.isnan(self.env.mine_ground_copy[row + i, column + j])):
                            self.env.click_square(row + i, column + j)
                            self.env.render_env()

    def _flag_all_neighbours(self, row, column):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (row + i >= 0 and column + j >= 0 and row + i < self.env.mine_ground_copy.shape[0]
                        and column + j < self.env.mine_ground_copy.shape[1] and (not self.env.flags[row + i, column + j])
                        and np.isnan(self.env.mine_ground_copy[row + i, column + j])):
                        self.env.add_mine_flag(row + i, column + j)
                        self.env.render_env()

    def _get_bomb(self, row, column):
        bomb_count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (row + i >= 0 and column + j >= 0 and row + i < self.env.mine_ground_copy.shape[0]
                        and column + j < self.env.mine_ground_copy.shape[1] and self.env.flags[row + i, column + j]):
                        bomb_count = bomb_count + 1
        return bomb_count

    def _get_unexplored(self, row, column):
        unexplored_count = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (row + i >= 0 and column + j >= 0 and row + i < self.env.mine_ground_copy.shape[0]
                        and column + j < self.env.mine_ground_copy.shape[1] and np.isnan(self.env.mine_ground_copy[row + i, column + j])):
                        unexplored_count = unexplored_count + 1
        return unexplored_count
