import torch

import cppex


# 我们使用c++定义的两个函数forward和backward定义了Function计算图节点，该节点接收两个输入，计算后给出一个输出
class TestFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        return cppex.forward(x, y)

    @staticmethod
    def backward(ctx, gradOutput):
        gradX, gradY = cppex.backward(gradOutput)
        return gradX, gradY


# 我们使用torch.nn.Module将将计算封装为标准的神经网络层
# 该神经网络层有一个可学习向量，与输入向量计算得到输出
class Test(torch.nn.Module):

    def __init__(self):
        super(Test, self).__init__()
        self.params = torch.nn.Parameter(torch.tensor([3, 4], dtype=torch.float32))

    def forward(self, inputA):
        return TestFunction.apply(inputA, self.params)


if __name__ == "__main__":
    t = Test()
    a = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
    z = t(a)
    sum_ = z.sum()
    sum_.backward()
    print(a.grad)
    # 上面的操作生成了一个计算图，计算图的节点是计算两个向量操作，输入是两个向量，其中一个向量是神经网络的可学习参数
    # 两个向量是leaf node，所以我们经过backward后，pytorch会调用accumulate函数进行leaf node节点的梯度计算
