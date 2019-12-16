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




train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size= 100, shuffle=True)



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



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)




    def make_layer(self, block, out_channels, blocks, stride=1):

        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):

            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for i in range(1, blocks):

            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.conv(x)

        out = self.bn(out)

        out= self.relu (out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out



batch_size = 100



net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2]
}
model = ResNet(**net_args)



# Cross Entropy Loss
error = nn.CrossEntropyLoss()

# Adam Optimizer
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



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

