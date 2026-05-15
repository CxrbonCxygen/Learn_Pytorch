# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_optim.py
# @Time : 2026/5/15 10:04

import torchvision
from torch import nn, optim  # 导入优化器模块
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 加载CIFAR-10测试数据集
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 创建数据加载器，batch_size=1表示每次只处理一个样本
dataloader = DataLoader(dataset, batch_size=1)


class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 卷积层：输入3通道，输出32通道，卷积核5x5，填充2
            MaxPool2d(2),  # 最大池化层：2x2窗口
            Conv2d(32, 32, 5, padding=2),  # 卷积层：输入32通道，输出32通道，卷积核5x5，填充2
            MaxPool2d(2),  # 最大池化层：2x2窗口
            Conv2d(32, 64, 5, padding=2),  # 卷积层：输入32通道，输出64通道，卷积核5x5，填充2
            MaxPool2d(2),  # 最大池化层：2x2窗口
            Flatten(),  # 展平层：将多维张量展平为一维向量
            Linear(1024, 64),  # 全连接层：输入1024维，输出64维
            Linear(64, 10)  # 全连接层：输入64维，输出10维（对应10个类别）
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 定义损失函数：交叉熵损失，常用于分类任务
loss_fn = nn.CrossEntropyLoss()
# 创建模型实例
model = CarbonOxygen()
# 定义优化器：使用SGD随机梯度下降，学习率为0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data  # 解包数据：imgs是图像数据，targets是真实标签
        output = model(imgs)  # 前向传播：将输入数据传入模型得到预测结果

        result_loss = loss_fn(output, targets)  # 计算损失：比较预测结果和真实标签

        optimizer.zero_grad()  # 清零梯度：清除上一次计算的梯度值
        result_loss.backward()  # 反向传播：计算当前损失对各参数的梯度
        optimizer.step()  # 参数更新：根据计算出的梯度更新模型参数
        running_loss = running_loss + result_loss
    print("第{}轮训练结束，总损失为{}".format(epoch + 1, running_loss))