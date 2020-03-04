import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch import FloatTensor
from torch.autograd import Variable

class Utils:

    def __init__(self):
        return

    def numpy_array_to_torch_tensor(self, numpy_array, dtype = np.float32, tensor_type = FloatTensor, is_volatile = False):
        """
        This function converts a numpy array to a pytorch tensor

        :param numpy_array (obj:`numpy array`): the numpy array to be converted
        :param dtype (obj:`numpy float type`): the dtype of the numpy array
        :param tensor_type (obj:`Pytorch Tensor`): the type of the final output tensor
        """

        if numpy_array.dtype != dtype:
            numpy_array = numpy_array.astype(dtype)
        return Variable(torch.from_numpy(numpy_array).type(tensor_type), volatile = is_volatile)
