# @Version : 1.0
# @Auther : CarbonOxygen
# @File : read_data.py
# @Time : 2026/4/28 20:47

from torch.utils.data import Dataset  # 导入 PyTorch 的数据集基类
from PIL import Image  # 导入 PIL 库用于读取和处理图片
import os  # 导入操作系统模块，用于文件路径操作

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        """
        初始化方法：创建数据集实例时自动调用
        参数:
            root_dir: 数据集根目录（如 dateset/train）
            label_dir: 类别文件夹名称（如 'ants' 或 'bees'）
        """
        self.root_dir = root_dir  # 保存根目录路径
        self.label_dir = label_dir  # 保存类别名称
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接完整类别路径
        self.img_path = os.listdir(self.path)  # 列出该类别下所有文件（图片）

    def __getitem__(self, idx):
        """
        获取单个样本：当使用 dataset[idx] 时自动调用
        参数:
            idx: 索引值（0 到 len(dataset)-1）
        返回:
            (img, label): 图片对象和对应标签的元组
        """
        img_name = self.img_path[idx]  # 根据索引获取第 idx 个图片的文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼接图片完整路径
        img = Image.open(img_item_path)  # 打开图片并加载到内存
        label = self.label_dir  # 设置标签为类别名称（如 'ants'）
        return img, label  # 返回图片和标签的元组

    def __len__(self):
        """
        获取数据集大小：当使用 len(dataset) 时自动调用
        返回:
            该类别下图片的总数量
        """
        return len(self.img_path)  # 返回文件列表的长度

root_dir = r'D:\Python_code\Learn_Pytorch\dateset\train'  # 定义训练集根目录（r 表示原始字符串，避免转义）
label_dir = 'ants_image'  # 指定要读取的类别为蚂蚁
bees_dir = 'bees_image'
ants_dataset = MyDataset(root_dir, label_dir)  # 创建 ants 数据集实例，自动执行 __init__
bees_dataset = MyDataset(root_dir, bees_dir)
train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))