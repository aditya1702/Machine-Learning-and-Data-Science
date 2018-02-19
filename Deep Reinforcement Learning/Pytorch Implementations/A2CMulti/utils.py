import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch import FloatTensor
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Utils:

    def __init__(self):
        return

    def numpy_array_to_torch_tensor(self, numpy_array, dtype = np.float32, tensor_type = FloatTensor):
        if numpy_array.dtype != dtype:
            numpy_array = numpy_array.astype(dtype)
        return Variable(torch.from_numpy(numpy_array).type(tensor_type))

    def torch_tensor_to_numpy_array(self, torch_tensor):
        return torch_tensor.data.squeeze(1).numpy()

    def _plot_environment_statistics(self, reward_per_episode):

        total_episodes = list(reward_per_episode.keys())
        total_rewards = list(reward_per_episode.values())
        plt.plot(total_episodes, total_rewards)
        plt.show()
