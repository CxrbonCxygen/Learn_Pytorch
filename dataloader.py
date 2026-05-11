# @Version : 1.0
# @Auther : CarbonOxygen
# @File : dataloader.py
# @Time : 2026/5/9 11:51
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download= True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中的第一张图片和标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("loader_logs")
step = 0
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, target = data
        # print(img.shape)
        # print(target)
        writer.add_images("epoch_test_data_"+str(epoch), img, step)
        step += 1

writer.close()