# import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from nn import *
import os
import scipy.io
import torch

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

train_data = scipy.io.loadmat('../data/data1/nist36_train_set1.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']

batch_size = 30

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self. num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

network = ConvNet()
optimizer = optim.Adam(network.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()


losslist=[]
acclist=[]


for epoch in range (100):

    total_loss = 0.0
    total_acc = 0.0

    for xb,yb in batches:
        xb= np.asarray(xb)
        xb= xb.reshape(30,1,32,32)
       # print (xb.shape)
        y_pred = network(torch.from_numpy(xb).float())

        ## Converting tensor back to np array

        y = y_pred.detach().numpy()

        accuracyvector = (np.argmax(yb, axis=1) == np.argmax(y, axis=1))
        total = np.sum(accuracyvector)
        total = total.astype(float)
        acc = total / np.size(accuracyvector)
        total_acc = total_acc + acc

        loss = criterion(y_pred, torch.max(torch.from_numpy(yb), 1)[1])

        total_loss = total_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_acc = total_acc / batch_num
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
