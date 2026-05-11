# @Version : 1.0
# @Auther : CarbonOxygen
# @File : nn_nodule.py
# @Time : 2026/5/9 15:14
import torch
from torch import nn

class CarbonOxygen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

carbonoxygen = CarbonOxygen()
x = torch.tensor(1.0)
print(type(x))
output = carbonoxygen(x)
print(output)