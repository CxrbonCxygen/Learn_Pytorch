# @Version : 1.0
# @Auther : CarbonOxygen
# @File : dataset_transform.py
# @Time : 2026/5/9 10:18

import torchvision
from torch.utils.tensorboard import SummaryWriter

from UsefulTransforms import writer

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(train_set[0])
# print(train_set.classes)
#
# img, label = test_set[0]
# print(img)
# print(label)
# print(train_set.classes[label])
#
# print(test_set[0])

writer = SummaryWriter("dataset_logs")
for i in range(10):
    img, label = test_set[i]
    writer.add_image("test_set", img, i)