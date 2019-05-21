from collections import deque

import numpy as np


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._running_mean = np.zeros(shape)
        self._running_standard_deviation = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._running_mean.shape
        self._n += 1
        if self._n == 1:
            self._running_mean[...] = x
        else:
            oldM = self._running_mean.copy()
            self._running_mean[...] = oldM + (x - oldM) / self._n
            self._running_standard_deviation[...] = self._running_standard_deviation + (x - oldM) * (x - self._running_mean)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._running_mean

    @property
    def var(self):
        return self._running_standard_deviation / (self._n - 1) if self._n > 1 else np.square(self._running_standard_deviation)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._running_mean.shape


class ZFilter:
    """
    y = (x - mean)/std
    using running estimates of mean and standard-deviation
    """

    def __init__(self, shape, calculate_mean = True, calculate_std = True, clip_value = 10.0):
        self.calculate_mean = calculate_mean
        self.calculate_std = calculate_std
        self.clip_value = clip_value

        self.running_variable = RunningStat(shape)

    def __call__(self, x, update = True):
        if update:
            self.running_variable.push(x)
        if self.calculate_mean:
            x = x - self.running_variable.mean
        if self.calculate_std:
            x = x / (self.running_variable.std + 1e-8)
        if self.clip_value:
            x = np.clip(x, -self.clip_value, self.clip_value)
        return x

    def output_shape(self, input_space):
        return input_space.shape
