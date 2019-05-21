import matplotlib
matplotlib.use('tkAgg')
from pylab import *
from matplotlib import colors
from utils.graph import Graph


class Environment():
    ProbabilityOfBlockedMaze = 0.4
    DimensionOfMaze = 10

    def __init__(self,
                 n = DimensionOfMaze,
                 p = ProbabilityOfBlockedMaze,
                 fire = None,
                 algorithm = None,
                 maze = None,
                 maze_copy = None,
                 colormesh = None):
        self.n = n
        self.p = p
        self.algorithm = algorithm
        self.maze = maze
        self.maze_copy = maze_copy
        self.colormesh = colormesh
        self.fire = fire
        self.counter = 0

        # The default colormap of our maze - 0: Black, 1: White, 2: Grey
        self.cmap = colors.ListedColormap(['black', 'white', 'grey', 'orange', 'red'])
        self.norm = colors.BoundaryNorm(boundaries = [0, 1, 2, 3, 4], ncolors = 4)

    def generate_maze(self, new_maze = None):

        if new_maze is not None:
            self.maze = new_maze
            self.original_maze = self.maze.copy()
            self.maze_copy = self.maze.copy()
            self.create_graph_from_maze()
            return

        self.maze = np.array([list(np.random.binomial(1, 1 - self.p, self.n)) for _ in range(self.n)])
        self.maze[0, 0] = 4
        self.maze[self.n - 1, self.n - 1] = 4

        if self.fire:
            self.maze[self.n - 1, 0] = 3

        # This will be the original maze
        self.original_maze = self.maze.copy()

        # Create a copy of maze to render and update
        self.maze_copy = self.maze.copy()

    def set_original_maze(self, new_maze):
        self.original_maze = new_maze

    def create_graph_from_maze(self):
        self.graph = Graph(maze = self.maze, algorithm = self.algorithm)
        self.graph.create_graph_from_maze()

    def render_maze(self, title = None, timer = 1e-15):

        # Create a mask for the particular cell and change its color to green
        masked_maze_copy = np.rot90(np.ma.masked_where(self.maze_copy == -1, self.maze_copy), k = 1)
        self.cmap.set_bad(color = 'green')

        # Plot the new maze
        if self.colormesh is None:
            self.colormesh = plt.pcolor(masked_maze_copy,
                                        cmap = self.cmap,
                                        norm = self.norm,
                                        edgecolor = 'k',
                                        linewidth = 0.5,
                                        antialiased = False)
        else:
            self.colormesh.set_array(masked_maze_copy.ravel())
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.show()
        plt.title(title)
        plt.pause(timer)

    def plot_maze(self, title = None, image_path = None):
        # Create a mask for the particular cell and change its color to green
        masked_maze_copy = np.rot90(np.ma.masked_where(self.maze_copy == -1, self.maze_copy), k = 9)
        self.cmap.set_bad(color = 'green')

        # Plot the new maze
        if self.colormesh is None:
            self.colormesh = plt.pcolor(masked_maze_copy,
                                        cmap = self.cmap,
                                        norm = self.norm,
                                        edgecolor = 'k',
                                        linewidth = 0.5,
                                        antialiased = False)
        else:
            self.colormesh.set_array(masked_maze_copy.ravel())
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        #plt.show()
        plt.title(title)
        plt.savefig(image_path)

    def update_color_of_cell(self, row, column):
        if self.maze[row, column] == 4:
            return
        self.maze_copy[row, column] = -1

    def reset_color_of_cell(self, row, column):
        if self.maze[row, column] == 4:
            return
        self.maze_copy[row, column] = 2

    def wild_fire(self, row, column):
        if (row == 0 and column == 0) or (row == self.n - 1 and column == self.n - 1):
            return
        self.maze_copy[row, column] = 3

    def reset_environment(self):
        self.maze = self.original_maze.copy()
        self.maze_copy = self.maze.copy()
        self.create_graph_from_maze()

    def modify_environment(self, row = None, column = None, new_maze = None):

        if new_maze is not None:
            self.maze = new_maze
        else:

            # If the cell's value is 1 change it to 0 and vice-versa
            if self.maze[row, column] == 0:
                self.maze[row, column] = 1
            else:
                self.maze[row, column] = 0

        # Update copy of maze
        self.maze_copy = self.maze.copy()
        self.create_graph_from_maze()
