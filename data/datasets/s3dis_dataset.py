"""
S3DIS数据集转换工具
将S3DIS场景数据集转换为物体分类数据集
"""

import os
import numpy as np
import h5py
import urllib.request
import tarfile
import zipfile
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import shutil

from .base_dataset import BaseDataset
from config import DatasetConfig, DatasetType, S3DIS_TO_SCANOBJECTNN_MAPPING


class S3DISDataset(BaseDataset):
    """S3DIS数据集转换器"""

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """
        初始化S3DIS数据集

        Args:
            config: 数据集配置
            split: 数据分割，可选 "train", "val", "test"
        """
        super().__init__(config, split)

        # S3DIS特定配置
        self.areas = config.s3dis_area
        self.s3dis_url = config.s3dis_url
        self.classes_to_include = config.s3dis_classes_to_include

        # 类别映射
        self.class_mapping = S3DIS_TO_SCANOBJECTNN_MAPPING

        # 设置类别信息（使用ScanObjectNN的类别）
        from config.dataset_config import SCANOBJECTNN_CLASSES, SCANOBJECTNN_CLASS_TO_ID, SCANOBJECTNN_ID_TO_CLASS
        self.class_names = SCANOBJECTNN_CLASSES
        self.class_to_id = SCANOBJECTNN_CLASS_TO_ID
        self.id_to_class = SCANOBJECTNN_ID_TO_CLASS

        # 加载数据
        self.load_data()

    def download(self) -> None:
        """
        下载S3DIS数据集
        """
        print("下载S3DIS数据集...")

        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)

        # S3DIS数据集URL
        s3dis_url = "http://buildingparser.stanford.edu/dataset/Stanford3dDataset_v1.2_Aligned_Version.zip"
        zip_path = os.path.join(self.data_dir, "Stanford3dDataset_v1.2_Aligned_Version.zip")

        print(f"从 {s3dis_url} 下载S3DIS数据集...")
        print("注意: S3DIS数据集约70GB，下载需要较长时间")

        try:
            # 显示下载进度
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = downloaded / total_size * 100
                print(f"下载进度: {percent:.1f}% ({downloaded / (1024**3):.1f} GB / {total_size / (1024**3):.1f} GB)",
                      end='\r')

            urllib.request.urlretrieve(s3dis_url, zip_path, report_progress)
            print(f"\n下载完成: {zip_path}")

            # 解压文件
            print("解压S3DIS数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            print(f"解压完成: {self.data_dir}")

            # 清理zip文件
            os.remove(zip_path)
            print("清理完成")

        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载S3DIS数据集:")
            print(f"1. 访问: http://buildingparser.stanford.edu/dataset.html")
            print(f"2. 下载: Stanford3dDataset_v1.2_Aligned_Version.zip")
            print(f"3. 解压到: {self.data_dir}")
            raise

    def preprocess(self) -> None:
        """
        预处理S3DIS数据集
        从场景中提取物体实例并构建分类数据集
        """
        print("预处理S3DIS数据集...")

        # 原始数据目录
        raw_data_dir = os.path.join(self.data_dir, "Stanford3dDataset_v1.2_Aligned_Version")

        if not os.path.exists(raw_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {raw_data_dir}")

        # 提取物体实例
        extracted_objects = self._extract_object_instances(raw_data_dir)

        if len(extracted_objects) == 0:
            raise ValueError("未提取到任何物体实例")

        print(f"共提取到 {len(extracted_objects)} 个物体实例")

        # 构建分类数据集
        self._build_classification_dataset(extracted_objects)

        print("S3DIS数据集预处理完成")

    def _extract_object_instances(self, raw_data_dir: str) -> List[Dict[str, Any]]:
        """
        从原始S3DIS数据中提取物体实例

        Args:
            raw_data_dir: 原始数据目录

        Returns:
            List[Dict[str, Any]]: 提取的物体实例列表
        """
        extracted_objects = []

        # S3DIS区域目录
        area_dirs = [d for d in os.listdir(raw_data_dir)
                     if d.startswith("Area_") and os.path.isdir(os.path.join(raw_data_dir, d))]

        # 只处理配置中指定的区域
        area_dirs = [d for d in area_dirs if int(d.split("_")[1]) in self.areas]

        print(f"处理区域: {area_dirs}")

        for area_dir in area_dirs:
            area_path = os.path.join(raw_data_dir, area_dir)
            room_dirs = [d for d in os.listdir(area_path)
                         if os.path.isdir(os.path.join(area_path, d))]

            for room_dir in room_dirs:
                room_path = os.path.join(area_path, room_dir)

                # 读取Annotations目录
                annotations_dir = os.path.join(room_path, "Annotations")
                if not os.path.exists(annotations_dir):
                    continue

                print(f"处理区域 {area_dir}, 房间 {room_dir}")

                # 读取每个物体文件
                obj_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]

                for obj_file in obj_files:
                    obj_path = os.path.join(annotations_dir, obj_file)

                    # 解析物体类别
                    class_name = obj_file.split('_')[0]

                    # 只处理指定的类别
                    if class_name not in self.classes_to_include:
                        continue

                    # 映射到ScanObjectNN类别
                    mapped_class = self.class_mapping.get(class_name)
                    if mapped_class is None:
                        continue

                    # 读取点云数据
                    try:
                        points = np.loadtxt(obj_path)
                        if points.shape[0] < 10:  # 跳过点太少的物体
                            continue

                        # 添加到提取列表
                        extracted_objects.append({
                            "points": points.astype(np.float32),
                            "class_name": mapped_class,
                            "area": area_dir,
                            "room": room_dir,
                            "object_file": obj_file,
                            "original_class": class_name
                        })

                    except Exception as e:
                        print(f"读取文件 {obj_path} 失败: {e}")
                        continue

        return extracted_objects

    def _build_classification_dataset(self, extracted_objects: List[Dict[str, Any]]) -> None:
        """
        构建分类数据集

        Args:
            extracted_objects: 提取的物体实例列表
        """
        # 按类别分组
        class_groups = {}
        for obj in extracted_objects:
            class_name = obj["class_name"]
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(obj)

        print("按类别统计:")
        for class_name, objects in class_groups.items():
            print(f"  {class_name}: {len(objects)} 个实例")

        # 保存数据集
        processed_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # 保存每个类别的数据
        all_points = []
        all_labels = []

        for class_name, objects in class_groups.items():
            class_id = self.class_to_id[class_name]

            for obj in objects:
                points = obj["points"]
                all_points.append(points)
                all_labels.append(class_id)

        # 转换为numpy数组
        all_points_array = np.array(all_points, dtype=object)
        all_labels_array = np.array(all_labels, dtype=np.int32)

        # 分割数据集
        n_samples = len(all_points_array)
        indices = np.arange(n_samples)
        np.random.seed(self.config.random_seed)
        np.random.shuffle(indices)

        train_end = int(n_samples * self.config.train_ratio)
        val_end = train_end + int(n_samples * self.config.val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # 保存各个分割
        splits = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }

        for split_name, split_indices in splits.items():
            split_points = [all_points_array[i] for i in split_indices]
            split_labels = all_labels_array[split_indices]

            # 保存为npz文件
            save_path = os.path.join(processed_dir, f"{split_name}_data.npz")
            np.savez_compressed(save_path,
                               points=split_points,
                               labels=split_labels)

            print(f"{split_name} 分割保存到: {save_path}")
            print(f"  - 样本数量: {len(split_points)}")
            print(f"  - 类别分布: {np.bincount(split_labels.flatten()).tolist()}")

        # 保存元数据
        metadata = {
            "total_samples": n_samples,
            "class_distribution": {cls: len(objs) for cls, objs in class_groups.items()},
            "areas_processed": self.areas,
            "classes_included": self.classes_to_include,
            "class_mapping": self.class_mapping,
            "split_ratios": {
                "train": self.config.train_ratio,
                "val": self.config.val_ratio,
                "test": self.config.test_ratio
            },
            "split_indices": {
                "train": train_indices.tolist(),
                "val": val_indices.tolist(),
                "test": test_indices.tolist()
            }
        }

        metadata_file = os.path.join(processed_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"元数据保存到: {metadata_file}")

    def _load_split_data(self) -> None:
        """
        加载指定分割的数据
        """
        processed_dir = os.path.join(self.data_dir, "processed")
        split_file = os.path.join(processed_dir, f"{self.split}_data.npz")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"分割数据文件不存在: {split_file}")

        # 加载数据
        data = np.load(split_file, allow_pickle=True)
        points = data['points']
        labels = data['labels'].flatten()

        # 转换为列表格式
        self.points = [points[i].astype(np.float32) for i in range(len(points))]
        self.labels = labels.tolist()

        print(f"加载 {self.split} 分割: {len(self.points)} 个样本")

        # 加载元数据
        metadata_file = os.path.join(processed_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元数据

        Returns:
            Dict[str, Any]: 元数据字典
        """
        return self.metadata

    def visualize_scene(self, area: str, room: str, save_path: Optional[str] = None) -> None:
        """
        可视化整个场景

        Args:
            area: 区域名称，如 "Area_1"
            room: 房间名称，如 "conferenceRoom_1"
            save_path: 保存路径，如果为None则显示图像
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 原始数据目录
        raw_data_dir = os.path.join(self.data_dir, "Stanford3dDataset_v1.2_Aligned_Version")
        room_path = os.path.join(raw_data_dir, area, room, "Annotations")

        if not os.path.exists(room_path):
            print(f"房间路径不存在: {room_path}")
            return

        # 读取所有物体
        all_points = []
        all_colors = []
        obj_files = [f for f in os.listdir(room_path) if f.endswith('.txt')]

        color_map = plt.cm.tab20

        for i, obj_file in enumerate(obj_files):
            obj_path = os.path.join(room_path, obj_file)
            try:
                points = np.loadtxt(obj_path)
                if points.shape[0] > 0:
                    all_points.append(points[:, :3])  # 只取XYZ
                    # 为每个物体分配颜色
                    color = color_map(i % 20)
                    all_colors.append(np.tile(color[:3], (points.shape[0], 1)))
            except Exception as e:
                print(f"读取文件 {obj_file} 失败: {e}")

        if not all_points:
            print("未找到有效数据")
            return

        # 合并所有点
        scene_points = np.vstack(all_points)
        scene_colors = np.vstack(all_colors)

        # 可视化
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2],
                  c=scene_colors, s=1, alpha=0.6)

        ax.set_title(f"S3DIS Scene: {area}/{room}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_s3dis_classification_dataset(config: DatasetConfig, force_reprocess: bool = False) -> None:
    """
    创建S3DIS分类数据集

    Args:
        config: 数据集配置
        force_reprocess: 是否强制重新处理
    """
    dataset = S3DISDataset(config, split="train")

    # 检查是否已经处理过
    processed_dir = os.path.join(config.data_dir, "processed")
    if os.path.exists(processed_dir) and not force_reprocess:
        print("S3DIS分类数据集已存在，跳过处理")
        return

    # 处理数据集
    dataset.preprocess()
    print("S3DIS分类数据集创建完成")


def test_s3dis():
    """测试S3DIS数据集转换"""
    from config import DatasetConfig

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.S3DIS,
        data_dir="./data/s3dis_test",
        num_points=1024,
        batch_size=4,
        s3dis_area=[1],  # 只测试Area 1以减少时间
        s3dis_classes_to_include=["table", "chair"],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    print("测试S3DIS数据集转换...")
    print("注意: 此测试需要S3DIS数据集，如果未下载会自动下载")
    print("由于数据集较大，测试可能需要较长时间")

    try:
        # 创建数据集（会自动下载和处理）
        print("创建S3DIS分类数据集...")
        create_s3dis_classification_dataset(config, force_reprocess=False)

        # 测试数据集加载
        print("测试数据集加载...")
        dataset = S3DISDataset(config, split="train")
        print(f"数据集大小: {len(dataset)}")
        print(f"类别数量: {len(dataset.class_names)}")

        # 获取一个样本
        sample = dataset[0]
        print(f"样本点云形状: {sample['points'].shape}")
        print(f"样本标签: {sample['label']} -> {dataset.id_to_class[sample['label'].item()]}")

        # 获取统计信息
        stats = dataset.get_statistics()
        print(f"数据集统计:")
        for key, value in stats.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: {type(value).__name__} ({len(value)} items)")
            else:
                print(f"  {key}: {value}")

        print("S3DIS数据集转换测试通过!")

    except Exception as e:
        print(f"测试失败: {e}")
        print("注意: 测试需要下载和处理S3DIS数据集，请确保有足够的磁盘空间和网络连接")


if __name__ == "__main__":
    test_s3dis()