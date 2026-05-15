# @Version : 1.0
# @Auther : CarbonOxygen
# @File : model.py
# @Time : 2026/5/15 15:50
import torch
from torch import nn

# 把神经网络保存到独立的文件中
class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()
        # 把神经网络搭建成Sequential
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
         )

    def forward(self, x):
        x = self.model(x)
        return x

# 在本文件测试网络模型正确性
if __name__ == '__main__':
    carbonoxygen = CarbonOxygen()
    input = torch.ones((64, 3, 32, 32))
    output = carbonoxygen(input)
    print(output.shape)