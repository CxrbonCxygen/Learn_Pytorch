# @Version : 1.0
# @Auther : CarbonOxygen
# @File : model_load.py
# @Time : 2026/5/15 15:06

import torch
import torchvision
from model_save import *

# 保存方式1：加载模型
model = torch.load("vgg16_method1.pth", weights_only= False)
print(model)
# 保存方式2：加载模型（参数）官方推荐 只保存权重
vgg16 = torchvision.models.vgg16(pretrained= False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16)

# 陷阱1
model = torch.load("carbonoxygen_model1.pth", weights_only= False)
print(model)