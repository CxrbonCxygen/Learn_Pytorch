# @Version : 1.0
# @Auther : CarbonOxygen
# @File : model_pretrained.py
# @Time : 2026/5/15 10:40
from torch import nn
import torchvision

# train_data = torchvision.datasets.ImageNet(root="./data_image_net", split="train", download=True, transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

# 将vgg16应用在CIFAR10上
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, download=True, transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_false)