# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_linear.py
# @Time : 2026/5/13 11:09
import torchvision
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download= True)
dataloader = DataLoader(dataset, batch_size=64)

# 写网络结构
class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

carbonoxygen = CarbonOxygen()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1,1,1,-1))
    # 将输入压缩成1维
    output = torch.flatten(imgs)
    print(output.shape)
    output = carbonoxygen(output)
    print(output.shape)