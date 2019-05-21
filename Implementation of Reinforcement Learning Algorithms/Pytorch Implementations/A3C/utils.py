import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch import FloatTensor
from torch.autograd import Variable

class Utils:

    def __init__(self):
        return

    def numpy_array_to_torch_tensor(self, numpy_array, dtype = np.float32):
        if numpy_array.dtype != dtype:
            numpy_array = numpy_array.astype(dtype)
        return Variable(torch.from_numpy(numpy_array).type(FloatTensor))

    def save_environment_info(self,
                               global_episode_counter,
                               global_episode_reward,
                               local_episode_reward,
                               worker_id,
                               reward_per_episode_queue):
        with global_episode_counter.get_lock():
            global_episode_counter.value += 1
        with global_episode_reward.get_lock():
            if global_episode_reward.value == 0.:
                global_episode_reward.value = local_episode_reward
            else:
                global_episode_reward.value = global_episode_reward.value * 0.99 + local_episode_reward * 0.01
        reward_per_episode_queue.put(global_episode_reward.value)
        print(
            worker_id,
            "Episode number:", global_episode_counter.value,
            "| Episode reward: %.0f" % global_episode_reward.value,
        )
