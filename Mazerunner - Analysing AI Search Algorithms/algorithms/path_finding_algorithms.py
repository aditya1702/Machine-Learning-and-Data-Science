import numpy as np
import queue as Q
from utils.environment import Environment


class PathFinderAlgorithm():
    DfsString = "dfs"
    BfsString = "bfs"
    AStarString = "astar"
    ThinningAStar = "thin_astar"
    FireString = "firealgo"

    def __init__(self, environment = None, algorithm = None, visual = False, heuristic = None, q = None):
        self.environment = environment
        self.graph_maze = self.environment.graph.graph_maze
        self.algorithm = algorithm
        self.visual = visual
        self.heuristic = heuristic
        self.visited = []
        self.path = []
        self.max_fringe_length = 0
        self.q = q

        if self.algorithm in ['dfs', 'bfs']:
            self.title = "Algorithm: " + self.algorithm
        else:
            self.title = "Algorithm: " + self.algorithm + "    Heuristic: " + self.heuristic

    def _get_unvisited_children(self, node_children):

        # If the algorithm is firealgo, then reorder children based on their heuristic values - distance from fire +
        # distance from destination
        if self.algorithm == "firealgo":
            temp_queue = Q.PriorityQueue()

        unvisited_children = []
        for child in node_children:
            if child is None:
                continue

            if child not in self.visited:
                if self.algorithm == "firealgo":
                    child.distance_from_fire = self._get_fire_distance(child)
                    temp_queue.put(child)
                else:
                    unvisited_children.append(child)

        if self.algorithm == "firealgo":
            unvisited_children = []
            while temp_queue.queue:
                unvisited_children.append(temp_queue.get())
            unvisited_children = unvisited_children[::-1]

        return unvisited_children

    def _get_final_path(self):
        node = self.graph_maze[self.environment.n - 1, self.environment.n - 1]
        while node is not None:
            self.path.append((node.row, node.column))
            node = node.parent

    def _get_euclidien_distance(self, node, dest):
        return np.sqrt((node.row - dest.row)**2 + (node.column - dest.column)**2)

    def _get_manhattan_distance(self, node, dest):
        return np.abs(node.row - dest.row) + np.abs(node.column - dest.column)

    def _calculate_heuristic(self, node, dest):
        if self.heuristic == "euclid":
            return self._get_euclidien_distance(node, dest)
        return self._get_manhattan_distance(node, dest)

    def _get_fire_distance(self, node):
        fire_blocks = np.argwhere(self.environment.maze_copy == 3)
        all_that_is_burning = []

        for i in zip(fire_blocks):
            all_that_is_burning.append(tuple((i[0][0], i[0][1])))

        time_taken_to_die = []

        for i in all_that_is_burning:
            temp = np.sqrt((node.row - i[0])**2 + (node.column - i[1])**2)
            time_taken_to_die.append(temp)

        time_before_i_call_fire_engine = min(time_taken_to_die)
        alpha_val = -0.5

        return (alpha_val * time_before_i_call_fire_engine)

    def get_final_path_length(self):
        return len(self.path)

    def get_number_of_nodes_expanded(self):
        return len(self.visited)

    def get_maximum_fringe_length(self):
        return self.max_fringe_length

    def _create_performance_metrics(self):
        self.performance_dict = dict()
        self.performance_dict['path_length'] = self.get_final_path_length()
        self.performance_dict['maximum_fringe_size'] = self.get_maximum_fringe_length()
        self.performance_dict['number_of_nodes_expanded'] = self.get_number_of_nodes_expanded()

    def _run_dfs(self, root, dest):

        self.fringe = [root]
        self.visited.append(root)
        while self.fringe:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            node = self.fringe.pop()

            # update color of the cell and render the maze
            if self.visual == True :            #Added visualisation parameter
                self.environment.update_color_of_cell(node.row, node.column)
                self.environment.render_maze(title = self.title)

            # if you reach the destination, then break
            if (node == dest):
                break

            if node not in self.visited:
                self.visited.append(node)

            # If there is no further path, then reset the color of the cell. Also, subsequently reset
            # the color of all parent cells along the path who have no other children to explore.
            flag = True
            while(flag):
                node_children = node.get_children(node = node, algorithm = self.algorithm)
                unvisited_children = self._get_unvisited_children(node_children)

                # If no unvisited children found, then reset the color of this cell in the current path
                # because there is no further path from this cell.
                if len(unvisited_children) == 0:
                    if self.visual == True:         #Added visualisation parameter --Nitin & Vedant
                        self.environment.reset_color_of_cell(node.row, node.column)
                        self.environment.render_maze(title = self.title)
                else:
                    for child in unvisited_children:
                        child.parent = node
                        self.fringe.append(child)
                    flag = False

                node = node.parent
                if node is None:
                    flag = False

    def _run_bfs(self, root, dest):

        self.fringe = [root]
        self.visited.append(root)
        while self.fringe:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            temp_path = []
            node = self.fringe.pop(0)

            if node not in self.visited:
                self.visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)
            unvisited_children = self._get_unvisited_children(node_children)

            for child in unvisited_children:

                # If child has been added to the fringe by some previous node, then dont add it again.
                if child not in self.fringe:
                    child.parent = node
                    self.fringe.append(child)

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while (flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above by popping the root first and the subsequent nodes.
            while (len(temp_path) != 0):
                temp_node = temp_path.pop()

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.update_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze(title = self.title)

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the entire path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze(title = self.title)

    def _run_astar(self, root, dest):

        # Root is at a distance of 0 from itself
        root.distance_from_source = 0

        self.fringe = Q.PriorityQueue()
        self.fringe.put(root)

        self.visited.append(root)
        while self.fringe.queue:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe.queue)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            temp_path = []
            node = self.fringe.get()

            if node not in self.visited:
                self.visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)

            for child in node_children:
                if child is None or child in self.visited:
                    continue

                if child not in self.fringe.queue:
                    child.parent = node
                    child.distance_from_dest = self._calculate_heuristic(child, dest)
                    child.distance_from_source = node.distance_from_source + 1
                    self.fringe.put(child)
                else:
                    if child.get_heuristic() >= node.distance_from_source + child.distance_from_dest:
                        child.parent = node
                        child.distance_from_source = node.distance_from_source + 1

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while (flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above by popping the root first and the subsequent nodes.
            while (len(temp_path) != 0):
                temp_node = temp_path.pop()

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.update_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the entire path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze()

    def _calculate_thinning_heuristic(self, simpler_maze_env, node_row, node_column):

        root = simpler_maze_env.graph.graph_maze[node_row, node_column]
        dest = simpler_maze_env.graph.graph_maze[simpler_maze_env.n - 1, simpler_maze_env.n - 1]

        # Root is at a distance of 0 from itself
        root.distance_from_source = 0

        fringe = Q.PriorityQueue()
        visited = []
        path = []
        fringe.put(root)

        visited.append(root)
        while fringe.queue:

            node = fringe.get()

            if node not in visited:
                visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)

            for child in node_children:
                if child is None or child in visited:
                    continue

                if child not in fringe.queue:
                    child.parent = node
                    child.distance_from_dest = self._calculate_heuristic(child, dest)
                    child.distance_from_source = node.distance_from_source + 1
                    fringe.put(child)
                else:
                    if child.get_heuristic() >= node.distance_from_source + child.distance_from_dest:
                        child.parent = node
                        child.distance_from_source = node.distance_from_source + 1

            # if you reach the destination, then break
            if (node == dest):
                break

        node = simpler_maze_env.graph.graph_maze[simpler_maze_env.n - 1, simpler_maze_env.n - 1]
        while node is not None:
            path.append((node.row, node.column))
            node = node.parent

        return len(path)

    def _create_simpler_maze(self, maze):
        zero_value_indices = list(zip(*(np.where(maze == 0)[0], np.where(maze == 0)[1])))
        zero_values_indices_length = range(len(zero_value_indices))
        random_zero_value_indices = np.random.choice(zero_values_indices_length,
                                                     size = int(self.q * len(zero_value_indices)),
                                                     replace = False)

        for index in random_zero_value_indices:
            row, column = zero_value_indices[index]
            maze[row, column] = 1

        # Set the source in the original maze to a open in the new simpler maze
        maze[0, 0] = 1

        return maze

    def _run_thinning_astar(self, root, dest):

        current_maze_copy = self.environment.maze.copy()
        simpler_maze = self._create_simpler_maze(maze = current_maze_copy)
        simpler_env = Environment(n = self.environment.n, p = self.environment.p)
        simpler_env.generate_maze(new_maze = simpler_maze)

        # Root is at a distance of 0 from itself
        root.distance_from_source = 0

        self.fringe = Q.PriorityQueue()
        self.fringe.put(root)

        self.visited.append(root)
        while self.fringe.queue:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe.queue)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            temp_path = []
            node = self.fringe.get()

            if node not in self.visited:
                self.visited.append(node)

            node_children = node.get_children(node = node, algorithm = self.algorithm)

            for child in node_children:
                if child is None or child in self.visited:
                    continue

                if child not in self.fringe.queue:
                    child.parent = node

                    temp_simpler_maze = simpler_maze.copy()
                    temp_simpler_maze[child.row, child.column] = 4
                    simpler_env.modify_environment(new_maze = temp_simpler_maze)

                    child.distance_from_dest = self._calculate_thinning_heuristic(simpler_maze_env = simpler_env,
                                                                                  node_row = child.row,
                                                                                  node_column = child.column)
                    child.distance_from_source = node.distance_from_source + 1
                    self.fringe.put(child)
                else:
                    if child.get_heuristic() >= node.distance_from_source + child.distance_from_dest:
                        child.parent = node
                        child.distance_from_source = node.distance_from_source + 1

            # Get the path through which you reach this node from the root node
            flag = True
            temp_node = node
            while (flag):
                temp_path.append(temp_node)
                temp_node = temp_node.parent
                if temp_node is None:
                    flag = False
            temp_path_copy = temp_path.copy()

            # Update the color of the path which we found above by popping the root first and the subsequent nodes.
            while (len(temp_path) != 0):
                temp_node = temp_path.pop()

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.update_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze(title = self.title)

            # if you reach the destination, then break
            if (node == dest):
                break

            # We reset the entire path again to render a new path in the next iteration.
            while (len(temp_path_copy) != 0):
                temp_node = temp_path_copy.pop(0)

                # Visualisation Parameter added
                if self.visual == True:
                    self.environment.reset_color_of_cell(temp_node.row, temp_node.column)
                    self.environment.render_maze(title = self.title)


    def _charizard(self):

        fire_blocks = np.argwhere(self.environment.maze_copy == 3)
        i_curr_burn = []

        for i in zip(fire_blocks):
            i_curr_burn.append(tuple((i[0][0], i[0][1])))

        for i in i_curr_burn:
            curr = self.environment.graph.graph_maze[i[0], i[1]]
            fire_kids = curr.get_children(node = curr, algorithm = self.algorithm)

            # fire_grandkids = []

            for beta in fire_kids:
                if beta is None:
                    continue

                every_kid = beta.get_children(node = beta, algorithm = self.algorithm)

                k = 0
                for kid in every_kid:
                    if kid is None:
                        continue
                    if kid.value == 3:
                        k += 1
                val = np.random.choice(2, 1, [(0.5**k, 1 - (0.5**k))])
                if val[0] == 1:
                    self.environment.wild_fire(beta.row, beta.column)

    def _update_fire_heuristic(self):
        for child in self.fringe.queue:
            child.distance_from_fire = self._get_fire_distance(child)

    def _run_from_fire(self):

        root = self.graph_maze[0, 0]
        dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1]

        # Assign distance from each node to the destination
        for row in range(len(self.environment.maze)):
            for column in range(len(self.environment.maze)):
                if self.environment.maze[row, column] == 0:
                    continue
                self.graph_maze[row, column].distance_from_dest = self._get_euclidien_distance(
                    self.graph_maze[row, column], dest)

        self.fringe = [root]
        self.visited.append(root)
        while self.fringe:

            # Keep track of maximum fringe length
            fringe_length = len(self.fringe)
            if fringe_length >= self.max_fringe_length:
                self.max_fringe_length = fringe_length

            node = self.fringe.pop()

            # update color of the cell and render the maze
            if self.visual == True:  # Added visualisation parameter
                self.environment.update_color_of_cell(node.row, node.column)
                self._charizard()
                self.environment.render_maze()

            # if you reach the destination, then break
            if (node == dest):
                break

            if node not in self.visited:
                self.visited.append(node)

            # If there is no further path, then reset the color of the cell. Also, subsequently reset
            # the color of all parent cells along the path who have no other children to explore.
            flag = True
            while (flag):
                node_children = node.get_children(node = node, algorithm = self.algorithm)
                unvisited_children = self._get_unvisited_children(node_children)

                # If no unvisited children found, then reset the color of this cell in the current path
                # because there is no further path from this cell.
                if len(unvisited_children) == 0:
                    if self.visual == True:  # Added visualisation parameter --Nitin & Vedant
                        self.environment.reset_color_of_cell(node.row, node.column)
                        self.environment.render_maze()
                else:
                    for child in unvisited_children:
                        child.parent = node
                        self.fringe.append(child)
                    flag = False

                node = node.parent
                if node is None:
                    flag = False

    def run_path_finder_algorithm(self):
        if self.algorithm == self.DfsString:
            self._run_dfs(root = self.graph_maze[0, 0],
                          dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1])
        elif self.algorithm == self.BfsString:
            self._run_bfs(root = self.graph_maze[0, 0],
                          dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1])
        elif self.algorithm == self.AStarString:
            self._run_astar(root = self.graph_maze[0, 0],
                            dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1])
        elif self.algorithm == self.FireString:
            self._run_from_fire()
        else:
            self._run_thinning_astar(root = self.graph_maze[0, 0],
                                     dest = self.graph_maze[self.environment.n - 1, self.environment.n - 1])

        # Get the final path
        self._get_final_path()

        # Create performance metrics
        self._create_performance_metrics()

        if len(self.path) == 1:
            print("NO PATH FOUND")
            return

        # Reverse the final saved path
        self.path = self.path[::-1]

        # Display the final highlighted path
        if self.visual == True:
            self.environment.render_maze(timer = 0.1)
