import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import os


os.environ['KMP_DUPLICATE_LIB_OK']= 'True'



## Loading MNIST dataset.

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size= 100, shuffle=True)




## Function to train the network


def train (train_loader, optimizer, network, criterion):
    loss_list = []
    iteration_list = []
    accuracy_list = []

    correct = 0
    total = 0
    count=0

    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        temp, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        accuracy = 100 * correct / float(total)

        loss_list.append(loss.item())
        iteration_list.append(count)
        accuracy_list.append(accuracy)

        print('Batch: {}  Loss: {}  Accuracy: {} %'.format(i, loss.item(), accuracy))


    return accuracy_list, loss_list



## Highway Network introduction and explanation in write-up


class HighwayNet(nn.Module):

    def __init__(self, num_layers=3, bias=-1, max_pool=True):

        super(HighwayNet, self).__init__()
        self.num_layers = num_layers

        self.max_pool = max_pool

        img_dim = 28

        self.conv = nn.Conv2d(1, 32, kernel_size= 3, padding= (1,1))


        self.conv_H = nn.ModuleList([ nn.Conv2d(32, 32, kernel_size= 3,padding= (1,1)) for _ in range(num_layers)])


        self.conv_T = nn.ModuleList([nn.Conv2d(32, 32, kernel_size= 3,padding= (1,1)) for _ in range(num_layers)])


        self.max_pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear( 32* int (img_dim/2) * int (img_dim/2), 64)

        self.fc2 = nn.Linear(64, 10)


        for i in range(num_layers):
            self.conv_T[i].bias.data.fill_(bias)



    def forward(self, x):
        x = F.relu(self.conv(x))

        for i in range(self. num_layers):

            H = F.relu(self.conv_H[i](x))
            T = torch.sigmoid(self.conv_T[i](x))
            x = H * T + x * (1 - T)
            x = self.dropout(x)

        if self.max_pool:
            x = self.max_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



# Cross Entropy Loss
error = nn.CrossEntropyLoss()

## Highway net Model
model = HighwayNet()

# Adam Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



## Training only for one epoch and plotting the train accuracy

for epoch in range (1):

    acclist,loss_list= train (train_loader, optimizer, model, error)

    plt.plot(loss_list, 'r')
    plt.xlabel('batches')
    plt.ylabel('loss')
    plt.show()
    #
    plt.plot(acclist, 'r')

    plt.xlabel('batches')
    plt.ylabel('accuracy')
    plt.show()

