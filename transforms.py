# @Version : 1.0
# @Auther : CarbonOxygen
# @File : transforms.py
# @Time : 2026/5/6 14:42
from PIL import Image
from sympy.tensor import tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import writer

# tensor数据类型
# 1.transforms该如何使用？
# 2.为什么我们需要使用tensor数据类型？

img_path = r"train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")


# 1.transforms该如何使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 直接用tensor输入，然后add_image()到tensorboard中
# 观察结果
writer.add_image("Tensor_img",tensor_img)
# 关闭
writer.close()

print(tensor_img)
