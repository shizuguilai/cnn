import torch
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
train_dataset = torchvision.datasets.ImageFolder('../picture', transform=transfrom)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size)

test_dataset = torchvision.dataset.ImageFolder('../test', transform=transfrom)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

"""
define model
"""
#first data = 32 * 640 * 480 (depth, width, height)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding= 1), # out = 8 * 640 * 480
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2), # out = 8* 320 * 240
            
            nn.Conv2d(8, 16, kernel_size=3, padding= 1), # out = 16 * 320 * 240
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2), # out = 16 * 160 * 120
            
            nn.Conv2d(16, 32, kernel_size=3, padding= 1), # out = 32 * 160 * 120
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2), #out = 32 * 80 * 60
            
            nn.Conv2d(32, 64, kernel_size=3, padding= 1), # out = 64 * 80 * 60
            nn.ReLU(),
            nn.MaxPool2d(stride = 2, kernel_size = 2))  # out = 64 * 40 * 30

        self.dense = nn.Sequential(
            nn.Linear(64 * 40 * 30, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 64 * 40 * 30)
        x = self.dense(x)
        return x

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

#study
num_epochs = 100
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
        if idx % batch_size == 0:
            print('epoch {}, batch{}, loss = {:g}'.format(epoch, idx, loss.item()))
			
#test
correct = 0
total = 0
for images, labels in test_loader:
	preds = model(images)
	predicted = torch.argmax(preds, 1)
	total += labels.size(0)
	correct += (predicted == labels).sum().item()

accuracy = correct  / total
print('correct: ',correct)
print('total: ', total)
print('accuracy: ',accuracy)

print('whether to  save or not? y/Y')
judge = input()
if judge == 'y' or judge == 'Y' :
    torch.save(model, 'model_100.pth')







