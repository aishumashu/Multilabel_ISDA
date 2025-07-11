import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset


class YOLOMultiLabelDataset(Dataset):
    """YOLO格式的多标签数据集"""

    def __init__(self, img_dir, label_dir, class_names_file, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # 读取类别名称
        with open(class_names_file, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_names)

        # 获取所有图片文件
        self.img_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))

        self.img_files = sorted(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        # 读取标签文件
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        # 创建多标签向量
        labels = torch.zeros(self.num_classes)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if 0 <= class_id < self.num_classes:
                            labels[class_id] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, labels
