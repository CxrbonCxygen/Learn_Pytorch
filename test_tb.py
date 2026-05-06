# @Version : 1.0
# @Auther : CarbonOxygen
# @File : test_tb.py
# @Time : 2026/4/29 10:31

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = r"train/ants_image/0013035.jpg"
image_path = r"train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))  # <class 'numpy.ndarray'>
print(img_array.shape)  # (512, 768, 3)

writer.add_image("test", img_array, 2, dataformats="HWC")
for i in range(100):
    writer.add_scalar("y = 3*x", 3*i, i)

writer.close()