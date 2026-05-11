# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_maxpool.py
# @Time : 2026/5/11 14:37
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download= True)

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

input = torch.reshape(input, (-1,1,5,5))
print(input.shape)

# 重写卷积核
class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode= False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

carbonoxygen = CarbonOxygen()
output = carbonoxygen(input)
print(output)