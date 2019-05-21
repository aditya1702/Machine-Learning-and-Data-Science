class Variable():
    def __init__(self,
                 value = None,
                 row = None,
                 column = None,
                 constraint_value = None,
                 has_mine = 1):
        self.value = value
        self.row = row
        self.column = column
        self.has_mine = has_mine
        self.constraint_equation = list()
        self.constraint_value = constraint_value
        self.neighbours = list()

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

    def __ne__(self, other):
        return self.row != other.row and self.column != other.column

    def __hash__(self):
        return hash((self.row, self.column))

    def add_constraint_variable(self, variable):
        self.constraint_equation.append(variable)

    def get_unopened_neighbours(self, env, use_probability_agent = False):
        unopened_neighbours = list()
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (self.row + i >= 0 and self.column + j >= 0 and self.row + i < env.n and self.column + j < env.n):

                    # If the neighbour has been opened or flagged, then continue
                    if env.clicked[self.row + i, self.column + j]:
                        continue

                    if use_probability_agent and env.clicked_and_not_revealed[self.row + i, self.column + j]:
                        continue

                    unopened_neighbours.append(env.variable_mine_ground_copy[self.row + i, self.column + j])
        return unopened_neighbours

    def get_flagged_mines(self, env):
        flagged_mines = 0

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if (i == 0 and j == 0):
                    continue
                if (self.row + i >= 0 and self.column + j >= 0 and self.row + i < env.n and self.column + j < env.n):
                    if env.flags[self.row + i, self.column + j]:
                        flagged_mines += 1

        return flagged_mines
