import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import os


os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size= 64, shuffle=True)

class SkipNet(nn.Module):
    def __init__(self):
        super(SkipNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1104, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(out)), 2))

        flat_x= x.view(-1,784 )

        out = out.view(-1, 320)

        out= torch.cat((flat_x,out),1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out)



network = SkipNet()
optimizer = optim.Adam(network.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()




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

            # Print Loss
        print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))


    return accuracy_list, loss_list


losslist=[]
acclist=[]

for epoch in range (1):


    acclist,loss_list= train (train_loader, optimizer, network, criterion)

    plt.plot(loss_list, 'r')
    plt.xlabel('batches')
    plt.ylabel('loss')
    plt.show()
    #
    plt.plot(acclist, 'r')

    plt.xlabel('batches')
    plt.ylabel('accuracy')
    plt.show()


