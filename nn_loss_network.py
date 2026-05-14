# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_loss_network.py
# @Time : 2026/5/14 20:47
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

loss = nn.CrossEntropyLoss()
carbonoxygen = CarbonOxygen()
for data in dataloader:
    imgs, targets = data
    output = carbonoxygen(imgs)
    # print(output)
    # print(targets)
    result_loss = loss(output, targets)
    # print(result_loss)
    # result_loss.backward() # 反向传播
    print("ok")