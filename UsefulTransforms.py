# @Version : 1.0
# @Auther : CarbonOxygen
# @File : UsefulTransforms.py
# @Time : 2026/5/7 12:32
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from test_tb import writer

writer = SummaryWriter("logs")

img = Image.open(r"images/蜡笔小新.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0]) # 没归一化前
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor) # 输入是tensor数据类型
print(img_norm[0][0][0]) # 归一化后
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size) # 没resize前
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor -> img_resize Tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    writer.add_image("RandomCrop", img_random, i)



writer.close()