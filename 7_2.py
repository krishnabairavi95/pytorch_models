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



## LeNet type architecture:



class LeNet (nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5,stride=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 17)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))

        x = self.maxpool2(x)

        x = x.view(-1, 16 * 5 * 5)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



squeeze_data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


lenet_data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


## For LeNet type model

trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform= lenet_data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size= 64, shuffle=True, num_workers=1)

net = LeNet()

optimizer = optim.Adam(net.parameters(), lr=0.001)


loss_func = torch.nn.CrossEntropyLoss()


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
        print('Batch: {}  Loss: {}  Accuracy: {} %'.format(i, loss.item(), accuracy))


    return accuracy_list, loss_list


# In case of LeNet type model

for epoch in range (30):


    acclist,loss_list= train (train_loader, optimizer, net, loss_func)

    # plt.plot(loss_list, 'r')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.show()
    # #
    # plt.plot(acclist, 'r')
    #
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.show()


### SqueezeNet



squeeze_trainset = datasets.ImageFolder(root='../data/oxford-flowers17/train', transform= squeeze_data_transform)

squeeze_train_loader = torch.utils.data.DataLoader(squeeze_trainset, batch_size= 64, shuffle=True, num_workers=1)


squeezenet = torchvision.models.squeezenet1_1(pretrained=True)

squeezenet.type(torch.FloatTensor)

squeezenet.num_classes = 17

final_conv = nn.Conv2d(512, squeezenet.num_classes, kernel_size=1)

squeezenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

squeeze_optimizer= torch.optim.Adam(squeezenet.classifier.parameters(), lr=1e-3)

squeezenet.type(torch.FloatTensor)

criterion = nn.CrossEntropyLoss().type(torch.FloatTensor)

print (squeezenet.parameters())


for params in squeezenet.parameters():
    params.requires_grad = False

for params in squeezenet.classifier.parameters():
    params.requires_grad = True



#### While running squeezenet please run this part


#
# for epoch in range (10):
#     print('Starting epoch %d / %d' % (epoch,10 ))
#
#     train(squeeze_train_loader, squeeze_optimizer, squeezenet, criterion)


