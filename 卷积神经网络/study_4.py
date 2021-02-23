import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader  
import torchvision.datasets
import torchvision.transforms
os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size = 100
#transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size)


"""
define model
"""
#first data = 32 * 640 * 480 (depth, width, height)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding= 1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding= 1), 
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2)) 

        self.dense = nn.Sequential(
            nn.Linear(128*14*14, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 128*14*14)
        x = self.dense(x)
        return x

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#study
num_epochs = 5
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        print('images.size: ', images.size())
        # print('images:',images)
        #print(labels)
        print('labels.size: ', labels.size())
        preds = model(images)
        print('preds:', preds)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('epoch {}, batch{}, loss = {:g}'.format(epoch, idx, loss.item()))

