import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.autograd.variable as Variable
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

path = "E:\学校的一些资料\文档\大二暑假\数据\鱼10组数据"#文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
#print("files: ",files)
txts = []
for file in files: #遍历文件夹
    print("file: ",file)
    position = path+'\\'+ file #构造绝对路径，"\\"，其中一个'\'为转义符
    files2 = os.listdir(position)
    for file2 in files2:
        #print("file2: ",file2)
        if(file2 == 'qp'):
            position2 = position + '\\' + file2
            files3 = os.listdir(position2)
            for file3 in files3:
                print("file3: ",file3)
                position3 = position2 + '\\' + file3
                with open(position3, 'r') as f:
                    data_list = f.readlines()
                    #print(data_list)
                    data_list = [i.split('\n')[0] for i in data_list]   #因为这个\n是在每行数据最后，所以split后分为第一个数字串和第二个空串，所以[0]
                    #print(data_list)
                    data_list = [i.split('\t') for i in data_list]
                    #print(data_list)  
                    data = [(float(i[0]), float(i[1])) for i in data_list]   #将字符转换为float
                    #print(data)  

                x = [i[0] for i in data]
                y = [i[1] for i in data]
                dev = max(y)
                y = [i / dev for i in y]
                plt.xlim(100, 3810)
                plt.ylim(0, 1.1)
                plt.plot(x, y, 'r-')
                #plt.legend(loc='best')  #图例是集中e于地图一角或一侧的地图上各种符号和颜色所代表内容与指标的说明，有助于更好的认识地图。
                subname = ''.join(filter(lambda x: x >= 'a' and x <= 'z', file3[0:-4]))
                #print('subname:', subname)
                name='picture/' + subname + '/' +  file3[0:-4] #保存文件
                isExists = os.path.exists('picture/' + subname)
                if not isExists:
                    os.makedirs('picture/' + subname)
                print("name: ", name)
                #plt.show()
                plt.savefig(name)
                plt.cla()


