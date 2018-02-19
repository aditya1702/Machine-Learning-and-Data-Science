import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data.dataloader as dataloader
from collections import OrderedDict

a = torch.rand(4)
b = torch.rand(4)

print torch.max(a, b ,out=a)
