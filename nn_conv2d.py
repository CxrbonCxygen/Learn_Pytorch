# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_conv2d.py
# @Time : 2026/5/9 16:07
from django.db.migrations import writer
from torch import nn
from torch.utils.data import DataLoader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from nn_nodule import carbonoxygen

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

carbonoxygen = CarbonOxygen()
print(carbonoxygen)

writer = SummaryWriter("cnov2d_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = carbonoxygen(imgs)
    print(imgs.shape)
    print(outputs.shape)


    # 报错Input should be a 4D numpy array with 3 channels
    writer.add_images("input", imgs, step)
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("output", outputs, step)
    step += 1


