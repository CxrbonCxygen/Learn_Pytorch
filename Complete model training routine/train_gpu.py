# @Version : 1.0
# @Auther : CarbonOxygen
# @File : train_gpu.py
# @Time : 2026/5/16 14:14

import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch.utils.data import DataLoader

# 准备训练数据集
train_data = torchvision.datasets.CIFAR10(
    root="../dataset",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10(
    root="../dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download= True
)

# 看看数据集有多少张图片
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size = 10 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
# 测试数据集的加载
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络（CIFAR10是10分类的，构建一个10分类的网络模型）
carbonoxygen = CarbonOxygen()
if torch.cuda.is_available():
    carbonoxygen = carbonoxygen.cuda()

# 创建损失函数：交叉熵损失，常用于分类任务
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 创建优化器：使用SGD随机梯度下降，学习率为0.01
learning_rate = 1e-2 # 等效0.01
optimizer = torch.optim.SGD(carbonoxygen.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 # 训练的次数
total_test_step = 0 # 测试的次数
epoch = 5 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("——————第 {} 轮训练开始——————".format(i+1))
    # 训练步骤开始
    carbonoxygen.train() # 训练模式
    for data in train_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = carbonoxygen(imgs) # 前向传播
            loss = loss_fn(output, targets) # 计算损失
            # 开始优化
            optimizer.zero_grad() # 优化器梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 优化器参数更新

            # 优化完毕，训练次数加一
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    carbonoxygen.eval() # 测试模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = carbonoxygen(imgs) # 前向传播
            loss = loss_fn(outputs, targets) # 计算损失
            total_test_loss += loss.item()
            # 对于图像分类有个Argmax方法，可以用来求出正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 每一轮保存模型
    torch.save(carbonoxygen, "carbonoxygen_{}.pth".format(i+1))
    # 模型保存方式2：
    torch.save(carbonoxygen.state_dict(), "carbonoxygen_{}.pth".format(i+1))
    print("模型已保存")

writer.close()