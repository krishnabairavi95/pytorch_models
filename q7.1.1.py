#

import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn as nn

import os


os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

train_data = scipy.io.loadmat('../data/nist36_train_set1.mat')


train_x, train_y = train_data['train_data'], train_data['train_labels']

input_length = len(train_x[0])

batch_size = 30
num_epochs= 100

batches = get_random_batches(train_x, train_y, batch_size)


model = nn.Sequential(nn.Linear(1024, 64),
                      nn.Sigmoid(),
                      nn.Linear(64, 36),
                      nn.Softmax())

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



loss_overall = []
accuracy_overall = []
for epoch in range(num_epochs):
    total_loss = 0
    total_acc = 0


    for xb, yb in batches:

        ## Converting np array to torch tensor

        y_pred = model(torch.from_numpy(xb).float())


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


    total_acc = total_acc / len(batches)
    loss_overall.append(total_loss)
    accuracy_overall.append(total_acc)

    print('epoch: ', epoch, ' loss: ', total_loss, 'accuracy: ', total_acc)


epoch_list=range (num_epochs)


plt.plot(epoch_list,loss_overall)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(epoch_list,accuracy_overall)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


