# @Version : 1.0
# @Auther : CarbonOxygen
# @File : hello_torch.py
# @Time : 2026/4/28 19:40

import torch

# 创建一个张量（你可以理解为一个数字矩阵）
x = torch.tensor([1.0, 2.0, 3.0])
print("我的第一个张量：", x)

# 验证GPU可用
print("GPU可用：", torch.cuda.is_available())

# 做一个简单的计算
y = x * 2
print("乘以2之后：", y)