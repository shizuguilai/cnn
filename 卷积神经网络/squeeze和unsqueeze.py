import torch

a = torch.tensor([1,2, 3, 4])

print('a:',a)
print('a.shape:',a.shape)

b = a.unsqueeze(1) #在参数输入的维度上加1的维度上增加一个维度

print('b:',b)
print('b.shape:', b.shape)

c = b.squeeze(1) #只有维度为1的时候才能去掉

print('c:',c)
print('c.shape:',c.shape)


#https://www.jb51.net/article/194848.htm

"""
squeeze的用法主要就是对数据的维度进行压缩或者解压。

先看torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，
比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。
squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。
还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。

再看torch.unsqueeze()这个函数主要是对数据维度进行扩充。
给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。
a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
"""