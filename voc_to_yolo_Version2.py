import os
import xml.etree.ElementTree as ET
from PIL import Image
import argparse


class VOCToYOLO:
    def __init__(self, voc_root, yolo_root, class_names=None):
        """
        初始化VOC到YOLO转换器

        Args:
            voc_root (str): VOC数据集根目录
            yolo_root (str): YOLO格式输出目录
            class_names (list): 类别名称列表，如果为None则自动从数据中提取
        """
        self.voc_root = voc_root
        self.yolo_root = yolo_root
        self.class_names = class_names if class_names else []
        self.class_to_id = {}

        # 创建输出目录结构
        os.makedirs(yolo_root, exist_ok=True)
        
        # 为每个数据集分割创建独立的目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(yolo_root, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(yolo_root, split, "labels"), exist_ok=True)

    def extract_classes(self):
        """从VOC数据集中提取所有类别"""
        if self.class_names:
            return

        classes = set()
        annotations_dir = os.path.join(self.voc_root, "Annotations")

        for xml_file in os.listdir(annotations_dir):
            if not xml_file.endswith(".xml"):
                continue

            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                classes.add(class_name)

        self.class_names = sorted(list(classes))
        print(f"发现 {len(self.class_names)} 个类别: {self.class_names}")

    def create_class_mapping(self):
        """创建类别名称到ID的映射"""
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        # 保存类别文件
        with open(os.path.join(self.yolo_root, "classes.txt"), "w") as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

    def convert_bbox(self, bbox, img_width, img_height):
        """
        将VOC格式的边界框转换为YOLO格式

        Args:
            bbox (tuple): (xmin, ymin, xmax, ymax)
            img_width (int): 图像宽度
            img_height (int): 图像高度

        Returns:
            tuple: (x_center, y_center, width, height) 归一化后的坐标
        """
        xmin, ymin, xmax, ymax = bbox

        # 计算中心点和宽高
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        return x_center, y_center, width, height

    def convert_annotation(self, xml_path, img_path):
        """
        转换单个XML标注文件

        Args:
            xml_path (str): XML文件路径
            img_path (str): 对应的图像文件路径

        Returns:
            list: YOLO格式的标注列表
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        # 验证图像尺寸（可选）
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                actual_width, actual_height = img.size
                if actual_width != img_width or actual_height != img_height:
                    print(f"警告: {img_path} 的实际尺寸与XML中记录的不符")
                    img_width, img_height = actual_width, actual_height

        annotations = []

        # 处理每个对象
        for obj in root.findall("object"):
            class_name = obj.find("name").text

            if class_name not in self.class_to_id:
                print(f"警告: 未知类别 '{class_name}'，跳过")
                continue

            # 检查是否为困难样本
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue  # 跳过困难样本

            # 获取边界框
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 转换为YOLO格式
            x_center, y_center, width, height = self.convert_bbox((xmin, ymin, xmax, ymax), img_width, img_height)

            class_id = self.class_to_id[class_name]
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return annotations

    def convert_dataset(self, split="train"):
        """
        转换整个数据集

        Args:
            split (str): 数据集分割 ('train', 'val', 'test')
        """
        # 读取分割文件
        split_file = os.path.join(self.voc_root, "ImageSets", "Main", f"{split}.txt")

        if not os.path.exists(split_file):
            print(f"警告: 分割文件 {split_file} 不存在")
            if split == "train":
                # 如果没有分割文件且是train，处理所有图像
                annotations_dir = os.path.join(self.voc_root, "Annotations")
                image_ids = [f.replace(".xml", "") for f in os.listdir(annotations_dir) if f.endswith(".xml")]
                print(f"将处理所有 {len(image_ids)} 张图像作为训练集")
            else:
                print(f"跳过 {split} 数据集")
                return
        else:
            with open(split_file, "r") as f:
                image_ids = [line.strip() for line in f.readlines()]

        print(f"开始转换 {split} 数据集，共 {len(image_ids)} 张图像")

        # 创建分割特定的输出目录
        split_images_dir = os.path.join(self.yolo_root, split, "images")
        split_labels_dir = os.path.join(self.yolo_root, split, "labels")

        converted_count = 0

        for image_id in image_ids:
            # 构建文件路径
            xml_path = os.path.join(self.voc_root, "Annotations", f"{image_id}.xml")

            # 查找对应的图像文件
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                potential_path = os.path.join(self.voc_root, "JPEGImages", f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break

            if not os.path.exists(xml_path):
                print(f"警告: 标注文件 {xml_path} 不存在，跳过")
                continue

            if img_path is None:
                print(f"警告: 图像文件 {image_id} 不存在，跳过")
                continue

            # 转换标注
            try:
                annotations = self.convert_annotation(xml_path, img_path)

                # 保存YOLO格式标注到对应的split目录
                yolo_label_path = os.path.join(split_labels_dir, f"{image_id}.txt")
                with open(yolo_label_path, "w") as f:
                    for annotation in annotations:
                        f.write(annotation + "\n")

                # 复制图像文件到对应的split目录
                yolo_img_path = os.path.join(split_images_dir, os.path.basename(img_path))
                if not os.path.exists(yolo_img_path):
                    import shutil
                    shutil.copy2(img_path, yolo_img_path)

                converted_count += 1

                if converted_count % 100 == 0:
                    print(f"已转换 {converted_count}/{len(image_ids)} 张图像")

            except Exception as e:
                print(f"转换 {image_id} 时出错: {e}")
                continue

        print(f"{split} 数据集转换完成！成功转换 {converted_count} 张图像")
        return converted_count

    def create_yaml_config(self, converted_splits):
        """
        创建YOLO格式的数据集配置文件

        Args:
            converted_splits (dict): 已转换的数据集分割及其图像数量
        """
        config = {
            "path": os.path.abspath(self.yolo_root),
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        # 根据实际转换的数据集添加路径
        if "train" in converted_splits and converted_splits["train"] > 0:
            config["train"] = "train/images"
        if "val" in converted_splits and converted_splits["val"] > 0:
            config["val"] = "val/images"
        if "test" in converted_splits and converted_splits["test"] > 0:
            config["test"] = "test/images"

        yaml_path = os.path.join(self.yolo_root, "dataset.yaml")

        with open(yaml_path, "w") as f:
            f.write(f"# YOLO dataset config\n")
            f.write(f"path: {config['path']}\n")
            
            if "train" in config:
                f.write(f"train: {config['train']}\n")
            if "val" in config:
                f.write(f"val: {config['val']}\n")
            if "test" in config:
                f.write(f"test: {config['test']}\n")
                
            f.write(f"\n# Classes\n")
            f.write(f"nc: {config['nc']}\n")
            f.write(f"names: {config['names']}\n")

        print(f"配置文件已保存到: {yaml_path}")
        
        # 打印数据集统计信息
        print("\n数据集统计:")
        for split, count in converted_splits.items():
            if count > 0:
                print(f"  {split}: {count} 张图像")


def main():
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC to YOLO format")
    parser.add_argument("--voc_root", type=str, required=True, help="VOC数据集根目录")
    parser.add_argument("--yolo_root", type=str, required=True, help="YOLO格式输出目录")
    parser.add_argument("--classes", type=str, nargs="+", help="类别名称列表")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="要转换的数据集分割")

    args = parser.parse_args()

    # 创建转换器
    converter = VOCToYOLO(args.voc_root, args.yolo_root, args.classes)

    # 提取类别
    converter.extract_classes()

    # 创建类别映射
    converter.create_class_mapping()

    # 转换数据集并记录转换结果
    converted_splits = {}
    for split in args.splits:
        count = converter.convert_dataset(split)
        converted_splits[split] = count if count is not None else 0

    # 创建配置文件
    converter.create_yaml_config(converted_splits)

    print("转换完成！")


if __name__ == "__main__":
    # 示例用法
    # python voc_to_yolo.py --voc_root /path/to/VOCdevkit/VOC2012 --yolo_root /path/to/yolo_dataset

    # 如果直接运行，可以使用以下示例
    voc_root = r"F:\dataset\VOCdevkit\VOC2007"  # 替换为您的VOC数据集路径
    yolo_root = r"F:\dataset\VOC_yolo"  # 替换为输出路径

    if len(os.sys.argv) == 1:  # 没有命令行参数时使用示例配置
        print("使用示例配置，请修改路径后运行")
        print("或使用命令行参数：")
        print("python voc_to_yolo.py --voc_root /path/to/VOCdevkit/VOC2012 --yolo_root /path/to/yolo_dataset")
        
        # 取消注释下面的代码来使用示例配置
        # converter = VOCToYOLO(voc_root, yolo_root)
        # converter.extract_classes()
        # converter.create_class_mapping()
        # converted_splits = {}
        # for split in ["train", "val"]:
        #     count = converter.convert_dataset(split)
        #     converted_splits[split] = count if count is not None else 0
        # converter.create_yaml_config(converted_splits)
        # print("转换完成！")
    else:
        main()