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

batch_size = 270
transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = torchvision.datasets.ImageFolder('E:\学习python的过程\picture', transform=transfrom)
train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size)


"""
define model
"""
#first data = 32 * 640 * 480 (depth, width, height)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding= 1), # out = 32 * 640 * 480
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2), # out = 32* 320 * 240
            nn.Conv2d(32, 64, kernel_size=3, padding= 1), # out = 64 * 320 * 240
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2)) #out = 64 * 160 * 120

        self.dense = nn.Sequential(
            nn.Linear(64 * 160 * 120, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 64 * 160 * 120)
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
        #print('images.size: ', images.size())
        # print('images:',images)
        #print(labels)
        #print('labels.size: ', labels.size())
        preds = model(images)
        #print('preds:', preds)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print('epoch {}, batch{}, loss = {:g}'.format(epoch, idx, loss.item()))

