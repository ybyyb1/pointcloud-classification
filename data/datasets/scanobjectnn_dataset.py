"""
ScanObjectNN数据集加载器
"""

import os
import numpy as np
import h5py
import urllib.request
import tarfile
import zipfile
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch

from .base_dataset import BaseDataset
from config import DatasetConfig, DatasetType, SCANOBJECTNN_CLASSES, SCANOBJECTNN_CLASS_TO_ID, SCANOBJECTNN_ID_TO_CLASS


class ScanObjectNNDataset(BaseDataset):
    """ScanObjectNN数据集"""

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """
        初始化ScanObjectNN数据集

        Args:
            config: 数据集配置
            split: 数据分割，可选 "train", "test"
        """
        super().__init__(config, split)

        # ScanObjectNN特定配置
        self.version = config.scanobjectnn_version  # "main_split" 或 "pb_t50_rs_split"
        self.data_url = config.scanobjectnn_url

        # 数据集文件路径
        self.h5_file = os.path.join(self.data_dir, f"{self.version}.h5")

        # 设置类别信息
        self.class_names = SCANOBJECTNN_CLASSES
        self.class_to_id = SCANOBJECTNN_CLASS_TO_ID
        self.id_to_class = SCANOBJECTNN_ID_TO_CLASS

        # 加载数据
        self.load_data()

    def download(self) -> None:
        """
        下载ScanObjectNN数据集
        """
        print(f"下载ScanObjectNN数据集 ({self.version})...")

        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)

        # ScanObjectNN数据集URL（示例，实际URL可能需要调整）
        base_url = "https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/"
        filename = f"{self.version}.h5"
        url = f"{base_url}{filename}"

        try:
            # 下载文件
            print(f"从 {url} 下载...")
            urllib.request.urlretrieve(url, self.h5_file)
            print(f"下载完成: {self.h5_file}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载ScanObjectNN数据集:")
            print(f"1. 访问: https://github.com/hkust-vgd/scanobjectnn")
            print(f"2. 下载文件: {filename}")
            print(f"3. 保存到: {self.h5_file}")
            raise

    def preprocess(self) -> None:
        """
        预处理ScanObjectNN数据集
        将数据转换为标准格式并保存
        """
        print("预处理ScanObjectNN数据集...")

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"数据文件不存在: {self.h5_file}")

        # 读取h5文件
        with h5py.File(self.h5_file, 'r') as f:
            # 获取数据
            train_points = f['train_points'][:]
            train_labels = f['train_labels'][:]
            test_points = f['test_points'][:]
            test_labels = f['test_labels'][:]

        # 保存为numpy格式以便快速加载
        train_save_path = os.path.join(self.data_dir, "train_data.npz")
        test_save_path = os.path.join(self.data_dir, "test_data.npz")

        np.savez_compressed(train_save_path,
                           points=train_points,
                           labels=train_labels)

        np.savez_compressed(test_save_path,
                           points=test_points,
                           labels=test_labels)

        print(f"训练数据保存到: {train_save_path}")
        print(f"测试数据保存到: {test_save_path}")

        # 保存统计信息
        stats = {
            "train_samples": len(train_points),
            "test_samples": len(test_points),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "train_class_distribution": np.bincount(train_labels.flatten()).tolist(),
            "test_class_distribution": np.bincount(test_labels.flatten()).tolist(),
        }

        import json
        stats_file = os.path.join(self.data_dir, "dataset_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"统计信息保存到: {stats_file}")

    def _load_split_data(self) -> None:
        """
        加载指定分割的数据
        """
        # 检查预处理数据是否存在
        train_file = os.path.join(self.data_dir, "train_data.npz")
        test_file = os.path.join(self.data_dir, "test_data.npz")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print("预处理数据不存在，开始预处理...")
            self.preprocess()

        # 加载数据
        if self.split == "train":
            data_file = train_file
        else:  # "test" 或 "val"
            data_file = test_file

        try:
            data = np.load(data_file)
            points = data['points']
            labels = data['labels'].flatten()  # 确保是一维数组
        except Exception as e:
            print(f"加载数据失败: {e}")
            print("尝试从原始h5文件加载...")
            self._load_from_h5()

        # 转换为列表格式
        self.points = [points[i].astype(np.float32) for i in range(len(points))]
        self.labels = labels.tolist()

        print(f"加载 {self.split} 分割: {len(self.points)} 个样本")

    def _load_from_h5(self) -> None:
        """
        从原始h5文件加载数据
        """
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"数据文件不存在: {self.h5_file}")

        with h5py.File(self.h5_file, 'r') as f:
            if self.split == "train":
                points = f['train_points'][:]
                labels = f['train_labels'][:].flatten()
            else:
                points = f['test_points'][:]
                labels = f['test_labels'][:].flatten()

        self.points = [points[i].astype(np.float32) for i in range(len(points))]
        self.labels = labels.tolist()

    def get_hard_split_indices(self) -> Dict[str, List[int]]:
        """
        获取困难样本的索引（如果可用）

        Returns:
            Dict[str, List[int]]: 困难样本索引
        """
        hard_split_file = os.path.join(self.data_dir, "hard_split_indices.npy")

        if not os.path.exists(hard_split_file):
            print("困难分割文件不存在")
            return {}

        try:
            indices = np.load(hard_split_file, allow_pickle=True).item()
            return indices
        except Exception as e:
            print(f"加载困难分割失败: {e}")
            return {}

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        归一化点云

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 归一化的点云
        """
        # 中心化
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # 缩放
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points

    def get_sample_by_class(self, class_name: str, n_samples: int = 1) -> List[Dict[str, Any]]:
        """
        获取指定类别的样本

        Args:
            class_name: 类别名称
            n_samples: 样本数量

        Returns:
            List[Dict[str, Any]]: 样本列表
        """
        if class_name not in self.class_to_id:
            raise ValueError(f"未知的类别: {class_name}")

        class_id = self.class_to_id[class_name]
        class_indices = [i for i, label in enumerate(self.labels) if label == class_id]

        if len(class_indices) == 0:
            return []

        # 随机选择样本
        selected_indices = np.random.choice(class_indices,
                                           size=min(n_samples, len(class_indices)),
                                           replace=False)

        samples = []
        for idx in selected_indices:
            samples.append(self[idx])

        return samples


def create_scanobjectnn_dataloader(config: DatasetConfig, split: str = "train",
                                   shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    创建ScanObjectNN数据加载器

    Args:
        config: 数据集配置
        split: 数据分割
        shuffle: 是否打乱数据

    Returns:
        DataLoader: PyTorch数据加载器
    """
    import torch

    dataset = ScanObjectNNDataset(config, split)

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )

    return dataloader


def test_scanobjectnn():
    """测试ScanObjectNN数据集"""
    from config import DatasetConfig

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.SCANOBJECTNN,
        data_dir="./data/scanobjectnn_test",
        num_points=1024,
        batch_size=4,
        scanobjectnn_version="main_split",
        scanobjectnn_url="https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/main_split"
    )

    print("测试ScanObjectNN数据集...")

    try:
        # 创建数据集
        dataset = ScanObjectNNDataset(config, split="train")
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

        # 测试数据加载器
        dataloader = create_scanobjectnn_dataloader(config, split="train", shuffle=True)
        batch = next(iter(dataloader))
        print(f"批次点云形状: {batch['points'].shape}")
        print(f"批次标签形状: {batch['label'].shape}")

        print("ScanObjectNN数据集测试通过!")

    except Exception as e:
        print(f"测试失败: {e}")
        print("注意: 测试需要下载数据集，请确保网络连接正常")


if __name__ == "__main__":
    test_scanobjectnn()