import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data.dataloader as dataloader


batch_size = 50.0
training_iters = 100000


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def main():
    train_loader = torch.utils.data.DataLoader(
                    MNIST('data',
                    train=True,
                    download=False,
                    transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                    MNIST('data',
                    train=False,
                    transform=transforms.ToTensor()),
                    batch_size=1000)

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    train_loss = []
    train_accu = []
    i = 0
    for epoch in range(15):
        for data, target in train_loader:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()    # calc gradients
            train_loss.append(loss.data[0])
            optimizer.step()   # update gradients
            prediction = output.data.max(1)[1]   # first column has actual prob.
            accuracy = prediction.eq(target.data).sum()/batch_size*100
            train_accu.append(accuracy)
            if i % 10 == 0:
                print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
            i += 1

main()
