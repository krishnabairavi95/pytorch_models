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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(out)), 2))
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out)



network = ConvNet()
optimizer = optim.Adam(network.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


def train (train_loader, optimizer, network, criterion):
    total_loss = 0.0
    total_acc = 0.0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = network(inputs)

        temp, predicted = torch.max(outputs, 1)
        out = (predicted == labels).squeeze()

        sum_acc = 0.0
        total = 0.0

        for each in out:
            total = total + 1.0
            sum_acc = sum_acc + each.item()

        acc = sum_acc / total
        total_acc = total_acc + acc

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

        return total_loss, total_acc



losslist=[]
acclist=[]

for epoch in range (100):

    total_loss, total_acc= train (train_loader, optimizer, network, criterion)
    #total_acc = total_acc / 10000
    losslist.append(total_loss)
    acclist.append(total_acc)
    print('epoch: ', epoch, ' loss: ', total_loss, 'accuracy: ', total_acc)



plt.plot(losslist,'r')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
#
plt.plot(acclist, 'r')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
# # #

