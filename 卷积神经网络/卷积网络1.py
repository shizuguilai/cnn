import torch
import numpy as np
import os
import time
import datetime
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   #网上搜的解决防止报错的方案


#读取数据函数,参数为数据的路径
def load(data_path):
    my_matrix = np.loadtxt(data_path ,dtype = float, usecols = [1], unpack = True)
    my_matrix = torch.tensor(my_matrix)
    #print("my_matrix: ",my_matrix)
    return my_matrix

#整个的读取,参数为path文件夹目录,names是要读取文件的名字数组
def read(path):
    flag = 1 #表示第一次，要创建一个张量t
    files= os.listdir(path) #得到文件夹下的所有文件名称
    for file1 in files: #遍历文件夹
        #print("file: ",file1)
        position = path+'\\'+ file1 #构造绝对路径，"\\"，其中一个'\'为转义符
        files2 = os.listdir(position)
        for file2 in files2:
            #print("file2: ",file2)
            position2 = position +'\\'+ file2
            #print("load(position2).shape: ",load(position2).shape)
            if flag == 1:
                #column = load(position2).shape
                #print("column: ", column)
                sum = load(position2).reshape(1, 1600)
                print("sum.shape: ",sum.shape)
                flag = 0
            else:
                sum = torch.cat([sum, load(position2).reshape(1, 1600)], 0)
    return sum


"""
操作部分
"""
#读取数据
sum_torch = read("E:\学校的一些资料\文档\大二暑假\数据\切片2020_12_19")  
#print("sum_torch:", sum_torch)

