import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.autograd.variable as Variable
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

with open('E:\学校的一些资料\文档\大三寒假，卷积，智能车\code-of-learn-deep-learning-with-pytorch-master\chapter3_NN\logistic-regression\data.txt', 'r') as f:
    data_list = f.readlines()
    #print(data_list)
    data_list = [i.split('\n')[0] for i in data_list]   #因为这个\n是在每行数据最后，所以split后分为第一个数字串和第二个空串，所以[0]
    #print(data_list)
    data_list = [i.split(',') for i in data_list]
    #print(data_list)  
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]   #将字符转换为float
    #print(data)         

x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]

plt.plot(plot_x0_0, plot_x0_1, 'ro', label = 'x_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label = 'x_1')
plt.legend(loc='best')  #图例是集中e于地图一角或一侧的地图上各种符号和颜色所代表内容与指标的说明，有助于更好的认识地图。
#plt.show()

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x

logistic_model = LogisticRegression()
if torch.cuda.is_available():
    print("模型存入gpu成功！！！！")
    logistic_model.cuda() #如果gpu可用，则存入gpu

criterion = nn.BCELoss()  #损失函数
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)  #梯度下降优化，第一个参数是模型的参数（logistic_model.parameters()）

x_data = [[i[0], i[1]] for i in data]
print(x_data)
y_data = [i[2] for i in data]
#print(y_data)
if torch.cuda.is_available():
    print('GPU计算！！！！！！！！！！！！')
    x = Variable(x_data).cuda()
    y = Variable(y_data).duda()
else:
    print('普通计算！！！！！！！！！！！！')
    x = Variable(x_data)
    y = Variable(y_data)
for epoch in range(5000):
    #forward
    out = logistic_model(x)
    loss = criterion(out, y)
    print_loss = loss.data[0]
    mask = out.ge(0.5).float()  #如果大于0.5就是1，小于0.5就是0
    correct = (abs(mask-y) < 0.001).sum()
    acc = correct.data[0] / x.size(0)
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print('*'*10)
        print('epoch{}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

#print
w0, w1 = logistic_model.lr.weight[0]
w0 = w0.data[0]
w1 = w1.data[0]
b = logistic_model.lr.bias.data[0]
plot_x = np.arange(30, 100, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x, plot_y)
plt.show()
        



