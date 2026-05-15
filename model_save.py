# @Version : 1.0
# @Auther : CarbonOxygen
# @File : model_save.py
# @Time : 2026/5/15 12:19

import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained =  False)
# 保存方式1
torch.save(vgg16, "vgg16_method1.pth")
# 保存方式2
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱1
class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
    def forward(self, x):
        x = self.conv1(x)
        return x

carbonoxygen = CarbonOxygen()
# 使用 torch.save(model, ...) 保存了完整模型结构
# 禁止加载自定义类以防恶意代码
torch.save(carbonoxygen, "carbonoxygen_model1.pth")