# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_ReLU.py
# @Time : 2026/5/13 09:41

import torch
import torchvision
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3]])


input = torch.reshape(input, (-1,1,2,2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download= True)
dataloader = DataLoader(dataset, batch_size=64)

class CarbonOxygen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

carbonoxygen = CarbonOxygen()
# output = carbonoxygen(input)
# print(output)

writer = SummaryWriter("relu_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = carbonoxygen(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
