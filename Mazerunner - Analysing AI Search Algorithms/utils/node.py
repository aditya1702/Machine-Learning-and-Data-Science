import numpy as np


class Node():
    def __init__(self,
                 algorithm = None,
                 value = None,
                 row = None,
                 column = None,
                 left = None,
                 right = None,
                 up = None,
                 down = None,
                 parent = None,
                 distance_from_dest = None,
                 distance_from_source = np.inf,
                 distance_from_fire = None,
                 num_nodes_before_this_node = None):
        self.algorithm = algorithm
        self.value = value
        self.row = row
        self.column = column
        self.parent = parent
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.distance_from_dest = distance_from_dest
        self.distance_from_source = distance_from_source
        self.distance_from_fire = distance_from_fire
        self.num_nodes_before_this_node = num_nodes_before_this_node
        self.distance_from_fire = None

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def __lt__(self, other):
        if self.algorithm == "firealgo":
            selfPriority = self.distance_from_fire + self.distance_from_dest
            otherPriority = other.distance_from_fire + other.distance_from_dest
        else:
            selfPriority = self.distance_from_source + self.distance_from_dest
            otherPriority = other.distance_from_source + other.distance_from_dest
        return selfPriority <= otherPriority

    def get_heuristic(self):
        if self.algorithm == "firealgo":
            return (self.distance_from_fire + self.distance_from_dest)
        else:
            return (self.distance_from_source + self.distance_from_dest)

    def get_children(self, node, algorithm):
        if algorithm == 'dfs':
            return [node.left, node.up, node.down, node.right]
        elif algorithm == 'bfs':
            return [node.right, node.down, node.up, node.left]
        else:
            return [node.left, node.up, node.down, node.right]
