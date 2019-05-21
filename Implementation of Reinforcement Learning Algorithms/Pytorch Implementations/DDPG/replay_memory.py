import random


class ReplayMemory():

    def __init__(self):
        self.replay_memory = list()

    def append(self, transition):
        self.replay_memory.append(transition)

    def pop(self):
        return self.replay_memory.pop()

    def sample(self, batch_size):
        return random.sample(self.replay_memory, batch_size)

    def get_size(self):
        return len(self.replay_memory)
