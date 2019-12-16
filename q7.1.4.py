import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import string
import os

from skimage import data

import matplotlib.pyplot as plt
import matplotlib
from nn import *
from q4 import *

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation


os.environ['KMP_DUPLICATE_LIB_OK']= 'True'


train_loader = torch.utils.data.DataLoader(datasets.EMNIST('../data', split='balanced', train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
               ])))

#
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the hh dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




network = ConvNet()
optimizer = optim.Adam(network.parameters(),lr= 0.001)
criterion = nn.CrossEntropyLoss()




def train (train_loader, optimizer, network, criterion, epoch):


    loss_list = []

    iteration_list = []

    accuracy_list = []

    correct = 0

    total = 0

    count = 0

    for i, (inputs, labels) in enumerate(train_loader):
        print (labels.shape)

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




# # #Please uncomment this part out to train


# for epoch in range(2):
#
#     train(train_loader, optimizer, network, criterion, epoch)
#
#     print ('done')
# torch.save(network, '../q7_1_4_weights')


model= torch.load('../q7_1_4_weights')


for img in os.listdir('../images'):

    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

    bboxes, bw = findLetters(im1)

    line=[]

    line.append(bboxes[0][:])

    all=[]

    output = ''

    for i in range(len(bboxes)-1):

        if bboxes[i+1][0]<bboxes[i][2]:

            line.append(bboxes[i+1][:])

        else:

            all.append(line)

            line = []

            line.append(bboxes[i+1][:])

    all.append(line)

    for i in range(len(all)):

        all[i].sort(key=lambda x: x[1])

    import string

    letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'])
    print('Image: {}'.format(img))


    output1 = ''

    for line in all:

        prev_end = line[0][3]

        for letter in line:

            img = bw[letter[0]:letter[2], letter[1]:letter[3]]

            if (letter[1] - prev_end) > 0.75 * (letter[3] - letter[1]):

                output1 += ' '


            img = np.pad(img, ((40, 40), (40, 40)), 'constant', constant_values=0.0)

            prev_end = letter[3]

            img = skimage.transform.resize(img, (28,28))

            img = skimage.morphology.dilation(img, skimage.morphology.square(3))

            input = img.T

            with torch.no_grad():
                output = model(torch.Tensor([[input]]))
                pred = output.max(1, keepdim=True)[1]
            output1 += letters[pred]
        output1 += '\n'


    print('Recognised text : \n{}'.format(output1))


