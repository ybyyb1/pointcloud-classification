"""
基础数据集类
定义点云数据集的通用接口
"""

import os
import abc
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from config import DatasetConfig


class BaseDataset(Dataset, abc.ABC):
    """点云数据集的抽象基类"""

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """
        初始化数据集

        Args:
            config: 数据集配置
            split: 数据分割，可选 "train", "val", "test"
        """
        super().__init__()
        self.config = config
        self.split = split
        self.data_dir = config.data_dir
        self.num_points = config.num_points

        # 数据缓存
        self.points = []  # 点云数据列表
        self.labels = []  # 标签列表
        self.class_names = []  # 类别名称列表
        self.class_to_id = {}  # 类别到ID的映射
        self.id_to_class = {}  # ID到类别的映射

        # 预处理变换
        self.transforms = None

        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    @abc.abstractmethod
    def download(self) -> None:
        """
        下载数据集
        子类必须实现此方法
        """
        pass

    @abc.abstractmethod
    def preprocess(self) -> None:
        """
        预处理数据集
        子类必须实现此方法
        """
        pass

    def load_data(self) -> None:
        """
        加载数据到内存
        子类可以重写此方法以实现自定义加载逻辑
        """
        # 检查数据是否存在
        if not self._check_data_exists():
            self.download()
            self.preprocess()

        # 加载具体的数据
        self._load_split_data()

    def _check_data_exists(self) -> bool:
        """
        检查数据是否已经存在

        Returns:
            bool: 数据是否存在
        """
        # 检查数据目录是否存在且非空
        if not os.path.exists(self.data_dir):
            return False

        # 检查是否有数据文件
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.npy', '.h5', '.pkl', '.ply'))]
        return len(data_files) > 0

    @abc.abstractmethod
    def _load_split_data(self) -> None:
        """
        加载指定分割的数据
        子类必须实现此方法
        """
        pass

    def __len__(self) -> int:
        """
        返回数据集大小

        Returns:
            int: 数据集样本数量
        """
        return len(self.points)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            Dict[str, Any]: 包含点云和标签的字典
        """
        points = self.points[idx].astype(np.float32)
        label = self.labels[idx]

        # 应用数据变换
        if self.transforms is not None:
            points = self.transforms(points)

        # 采样到固定点数
        points = self._sample_points(points)

        # 转换为torch张量
        points_tensor = torch.from_numpy(points).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "points": points_tensor,
            "label": label_tensor,
            "index": idx,
            "num_points": points.shape[0]
        }

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """
        采样点云到固定点数

        Args:
            points: 原始点云，形状为 (N, 3)

        Returns:
            np.ndarray: 采样后的点云，形状为 (num_points, 3)
        """
        n_points = points.shape[0]

        if n_points == self.num_points:
            return points
        elif n_points > self.num_points:
            # 随机采样
            indices = np.random.choice(n_points, self.num_points, replace=False)
            return points[indices]
        else:
            # 重复采样
            indices = np.random.choice(n_points, self.num_points, replace=True)
            return points[indices]

    def get_class_distribution(self) -> Dict[str, int]:
        """
        获取类别分布

        Returns:
            Dict[str, int]: 每个类别的样本数量
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        distribution = {}

        for label, count in zip(unique_labels, counts):
            class_name = self.id_to_class.get(label, f"class_{label}")
            distribution[class_name] = int(count)

        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        if len(self.points) == 0:
            return {}

        # 计算点云统计信息
        all_points = np.vstack(self.points)
        mean = np.mean(all_points, axis=0)
        std = np.std(all_points, axis=0)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        # 计算每个样本的点数
        num_points_per_sample = [p.shape[0] for p in self.points]

        return {
            "num_samples": len(self.points),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "class_distribution": self.get_class_distribution(),
            "points_mean": mean.tolist(),
            "points_std": std.tolist(),
            "points_min": min_vals.tolist(),
            "points_max": max_vals.tolist(),
            "avg_points_per_sample": np.mean(num_points_per_sample),
            "min_points_per_sample": np.min(num_points_per_sample),
            "max_points_per_sample": np.max(num_points_per_sample),
        }

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15, random_seed: int = 42) -> Dict[str, List[int]]:
        """
        分割数据集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子

        Returns:
            Dict[str, List[int]]: 包含各个分割索引的字典
        """
        # 验证比例总和为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例总和必须为1，当前为{total_ratio}")

        n_samples = len(self.points)
        indices = np.arange(n_samples)

        # 设置随机种子
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        # 计算分割点
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist()
        }

    def save_statistics(self, filepath: str) -> None:
        """
        保存统计信息到文件

        Args:
            filepath: 文件路径
        """
        import json

        stats = self.get_statistics()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def visualize_sample(self, idx: int, save_path: Optional[str] = None) -> None:
        """
        可视化样本

        Args:
            idx: 样本索引
            save_path: 保存路径，如果为None则显示图像
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        sample = self[idx]
        points = sample["points"].numpy()
        label = sample["label"].item()
        class_name = self.id_to_class.get(label, f"class_{label}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=points[:, 2], cmap='viridis', s=10, alpha=0.8)

        ax.set_title(f"Sample {idx}: {class_name} (Label: {label})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 设置相等的比例
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max()
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # 测试基类
    from config import DatasetConfig

    config = DatasetConfig(
        data_dir="./test_data",
        num_points=1024
    )

    # 创建一个简单的测试数据集类
    class TestDataset(BaseDataset):
        def download(self):
            print("下载测试数据...")
            # 创建一些随机数据
            np.random.seed(42)
            self.points = [np.random.randn(2000, 3) for _ in range(10)]
            self.labels = np.random.randint(0, 5, 10).tolist()
            self.class_names = [f"class_{i}" for i in range(5)]
            self.class_to_id = {cls: i for i, cls in enumerate(self.class_names)}
            self.id_to_class = {i: cls for i, cls in enumerate(self.class_names)}

        def preprocess(self):
            print("预处理测试数据...")

        def _load_split_data(self):
            print(f"加载 {self.split} 分割数据...")

    # 测试数据集
    dataset = TestDataset(config, split="train")
    dataset.load_data()

    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(dataset.class_names)}")

    # 获取一个样本
    sample = dataset[0]
    print(f"样本点云形状: {sample['points'].shape}")
    print(f"样本标签: {sample['label']}")

    # 获取统计信息
    stats = dataset.get_statistics()
    print(f"数据集统计: {stats}")