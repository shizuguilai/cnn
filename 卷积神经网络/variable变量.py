import torch
import torch.autograd.variable as Variable

#creat variable
x = Variable(torch.Tensor([1]), requires_grad = True)
w = Variable(torch.Tensor([2]), requires_grad = True)
b = Variable(torch.Tensor([3]), requires_grad = True)

#build a comptational graph
y = w * x + 3 * b

#compute gradients 计算梯度，其实就是对y中的每个变量进行求导，比如对b就是3，对w就是x，对x就是w
y.backward(torch.FloatTensor([0.1]))    #如果这个当中没有参数那么其实它的默认参数就是torch.FloatTensor([1]) ,中间的数字就是让求导的结果乘以它

#print out the gradients
print(x.grad)
print(w.grad)
print(b.grad)