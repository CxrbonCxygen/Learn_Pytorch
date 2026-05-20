# @Version : 1.0
# @Auther : CarbonOxygen
# @File : test.py
# @Time : 2026/5/16 16:48
import torch
import torchvision
from torch import nn
from PIL import Image

image_path = "./imgs/dog.jpg"
image = Image.open(image_path)
print(image)
# 调整图片尺寸
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)
print(image.shape)

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

# 加载网络模型
model = CarbonOxygen ()
model.load_state_dict(torch.load("carbonoxygen_10.pth"))
model.eval()
image = image.unsqueeze(0)
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# 获取预测结果并转换为 Python 整数
predicted_idx = output.argmax(1).item()
predicted_class = classes[predicted_idx]

print(f"模型输出的原始张量: {output}")
print(f"预测的索引: {predicted_idx}")
print(f"预测的类别 (Tag): {predicted_class}")