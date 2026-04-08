"""
Stanford3D数据集加载器
从Stanford3dDataset_v1.2构建物体分类数据集
"""

import os
import sys
import numpy as np
import h5py
import urllib.request
import zipfile
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path
import shutil
import glob

from .base_dataset import BaseDataset
from config import DatasetConfig, DatasetType


def is_kaggle_environment() -> bool:
    """检查是否在Kaggle环境中运行"""
    return os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def find_kaggle_input_file() -> Optional[str]:
    """
    在Kaggle输入目录中查找Stanford3D数据集文件

    Returns:
        Optional[str]: 找到的目录路径，如果未找到则返回None
    """
    if not is_kaggle_environment():
        return None

    # 可能的Kaggle输入目录
    possible_dirs = [
        '/kaggle/input/stanford3ddataset',
        '/kaggle/input/stanford-3d-dataset',
        '/kaggle/input/stanford3d',
        '/kaggle/input/s3dis',
        '/kaggle/input/stanford3ddataset-v1-2',
    ]

    for input_dir in possible_dirs:
        if os.path.exists(input_dir):
            print(f"在Kaggle输入目录中找到Stanford3D数据集: {input_dir}")
            return input_dir

    return None


class Stanford3DDataset(BaseDataset):
    """Stanford3D数据集"""

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """
        初始化Stanford3D数据集

        Args:
            config: 数据集配置
            split: 数据分割，可选 "train", "val", "test"
        """
        super().__init__(config, split)

        # Stanford3D特定配置
        self.areas = config.stanford3d_areas if hasattr(config, 'stanford3d_areas') else [1, 2, 3, 4, 5, 6]
        self.classes_to_include = config.stanford3d_classes_to_include if hasattr(config, 'stanford3d_classes_to_include') else []

        # 设置类别信息
        from config.dataset_config import STANFORD3D_CLASSES, STANFORD3D_CLASS_TO_ID, STANFORD3D_ID_TO_CLASS
        self.class_names = STANFORD3D_CLASSES
        self.class_to_id = STANFORD3D_CLASS_TO_ID
        self.id_to_class = STANFORD3D_ID_TO_CLASS

        # 如果未指定要包含的类别，则包含所有类别
        if not self.classes_to_include:
            self.classes_to_include = self.class_names

        # 原始数据目录 - 首先检查配置的数据目录中是否存在
        self.raw_data_dir = os.path.join(self.data_dir, "Stanford3dDataset_v1.2")
        # 如果不存在，检查上一级目录（用户可能已经将数据放在data/Stanford3dDataset_v1.2）
        if not os.path.exists(self.raw_data_dir):
            parent_raw = os.path.join(os.path.dirname(self.data_dir), "Stanford3dDataset_v1.2")
            if os.path.exists(parent_raw):
                self.raw_data_dir = parent_raw
                print(f"使用上级目录中的原始数据: {self.raw_data_dir}")

        # 处理后的数据目录
        self.processed_dir = os.path.join(self.data_dir, "processed")

        # 加载数据
        self.load_data()

    def download(self) -> None:
        """
        下载Stanford3D数据集
        """
        print("下载Stanford3D数据集...")

        # 检查原始数据是否已经存在
        if os.path.exists(self.raw_data_dir):
            # 检查是否有Area目录
            area_dirs = [d for d in os.listdir(self.raw_data_dir)
                         if d.startswith("Area_") and os.path.isdir(os.path.join(self.raw_data_dir, d))]
            if area_dirs:
                print(f"原始数据已存在: {self.raw_data_dir}，跳过下载")
                return

        # 首先检查是否在Kaggle输入目录中已存在数据集
        kaggle_input_dir = find_kaggle_input_file()
        if kaggle_input_dir:
            print(f"使用Kaggle输入目录中的数据集: {kaggle_input_dir}")
            # 复制到目标位置
            if not os.path.exists(self.raw_data_dir):
                shutil.copytree(kaggle_input_dir, self.raw_data_dir)
                print(f"数据集已复制到: {self.raw_data_dir}")
            return

        # 其次检查是否在Kaggle环境中，尝试通过Kaggle API下载
        if is_kaggle_environment():
            try:
                print("检测到Kaggle环境，尝试从Kaggle数据集下载...")
                self._download_from_kaggle()
                return
            except Exception as e:
                print(f"Kaggle下载失败，将尝试备用URL: {e}")

        # 备用方案：从原始URL下载
        print("尝试从原始URL下载Stanford3D数据集...")
        self._download_from_original_url()

    def _download_from_kaggle(self) -> None:
        """
        从Kaggle数据集下载
        """
        try:
            # 导入kaggle API
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                print("Kaggle API未安装，尝试安装...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                from kaggle.api.kaggle_api_extended import KaggleApi

            # 初始化API
            api = KaggleApi()
            try:
                api.authenticate()
            except Exception as auth_error:
                print(f"Kaggle API认证失败，尝试继续下载（公开数据集可能不需要认证）: {auth_error}")

            # Kaggle数据集名称
            kaggle_dataset = "sankalpsagar/stanford3ddataset"

            print(f"从Kaggle数据集 {kaggle_dataset} 下载...")

            # 下载数据集
            api.dataset_download_files(kaggle_dataset, path=self.data_dir, unzip=True, force=True, quiet=True)

            print(f"Kaggle下载成功: {self.data_dir}")

        except Exception as e:
            print(f"Kaggle下载失败: {e}")
            print("请确保已安装kaggle API: pip install kaggle")
            print("并配置了Kaggle API密钥")
            raise

    def _download_from_original_url(self) -> None:
        """
        从原始URL下载Stanford3D数据集
        """
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)

        # Stanford3D数据集URL
        stanford3d_url = "http://buildingparser.stanford.edu/dataset/Stanford3dDataset_v1.2_Aligned_Version.zip"
        zip_path = os.path.join(self.data_dir, "Stanford3dDataset_v1.2_Aligned_Version.zip")

        print(f"从 {stanford3d_url} 下载Stanford3D数据集...")
        print("注意: Stanford3D数据集约70GB，下载需要较长时间")

        try:
            # 显示下载进度
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = downloaded / total_size * 100
                print(f"下载进度: {percent:.1f}% ({downloaded / (1024**3):.1f} GB / {total_size / (1024**3):.1f} GB)",
                      end='\r')

            urllib.request.urlretrieve(stanford3d_url, zip_path, report_progress)
            print(f"\n下载完成: {zip_path}")

            # 解压文件
            print("解压Stanford3D数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            # 重命名解压后的目录（如果需要）
            extracted_dir = os.path.join(self.data_dir, "Stanford3dDataset_v1.2_Aligned_Version")
            if os.path.exists(extracted_dir) and not os.path.exists(self.raw_data_dir):
                os.rename(extracted_dir, self.raw_data_dir)

            print(f"解压完成: {self.raw_data_dir}")

            # 清理zip文件
            os.remove(zip_path)
            print("清理完成")

        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载Stanford3D数据集:")
            print(f"1. 访问: http://buildingparser.stanford.edu/dataset.html")
            print(f"2. 下载: Stanford3dDataset_v1.2_Aligned_Version.zip")
            print(f"3. 解压到: {self.data_dir}")
            print(f"4. 确保目录结构: {self.raw_data_dir}/Area_1/...")
            raise

    def preprocess(self) -> None:
        """
        预处理Stanford3D数据集
        从场景中提取物体实例并构建分类数据集
        """
        print("预处理Stanford3D数据集...")

        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"原始数据目录不存在: {self.raw_data_dir}")

        # 提取物体实例
        extracted_objects = self._extract_object_instances(self.raw_data_dir)

        if len(extracted_objects) == 0:
            raise ValueError("未提取到任何物体实例")

        print(f"共提取到 {len(extracted_objects)} 个物体实例")

        # 构建分类数据集
        self._build_classification_dataset(extracted_objects)

        print("Stanford3D数据集预处理完成")

    def _extract_object_instances(self, raw_data_dir: str) -> List[Dict[str, Any]]:
        """
        从原始Stanford3D数据中提取物体实例

        Args:
            raw_data_dir: 原始数据目录

        Returns:
            List[Dict[str, Any]]: 提取的物体实例列表
        """
        extracted_objects = []

        # Stanford3D区域目录
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

                    # 检查类别是否在定义的类别列表中
                    if class_name not in self.class_names:
                        print(f"警告: 类别 '{class_name}' 不在定义的类别列表中，跳过")
                        continue

                    # 读取点云数据
                    try:
                        points = np.loadtxt(obj_path)
                        if points.shape[0] < 10:  # 跳过点太少的物体
                            continue

                        # 只使用XYZ坐标，忽略RGB
                        if points.shape[1] >= 3:
                            points = points[:, :3]

                        # 添加到提取列表
                        extracted_objects.append({
                            "points": points.astype(np.float32),
                            "class_name": class_name,
                            "area": area_dir,
                            "room": room_dir,
                            "object_file": obj_file,
                            "num_points": points.shape[0]
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

        # 创建处理后的目录
        os.makedirs(self.processed_dir, exist_ok=True)

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
            split_points = all_points_array[split_indices]
            split_labels = all_labels_array[split_indices]

            # 保存为npz文件
            save_path = os.path.join(self.processed_dir, f"{split_name}_data.npz")
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

        metadata_file = os.path.join(self.processed_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"元数据保存到: {metadata_file}")

        # 也保存为HDF5格式以便与ScanObjectNN兼容
        self._save_as_h5(all_points_array, all_labels_array, train_indices, val_indices, test_indices)

    def _save_as_h5(self, all_points: np.ndarray, all_labels: np.ndarray,
                   train_indices: np.ndarray, val_indices: np.ndarray,
                   test_indices: np.ndarray) -> None:
        """
        保存为HDF5格式以便与ScanObjectNN兼容

        Args:
            all_points: 所有点云数据
            all_labels: 所有标签
            train_indices: 训练集索引
            val_indices: 验证集索引
            test_indices: 测试集索引
        """
        h5_file = os.path.join(self.processed_dir, "stanford3d_dataset.h5")

        # 准备数据
        train_points = [all_points[i] for i in train_indices]
        train_labels = all_labels[train_indices]

        # 合并验证集和测试集作为测试集（与ScanObjectNN格式一致）
        test_val_indices = np.concatenate([val_indices, test_indices])
        test_points = [all_points[i] for i in test_val_indices]
        test_labels = all_labels[test_val_indices]

        # 将点云填充/采样到统一大小
        target_num_points = 1024
        train_points_padded = []
        test_points_padded = []

        for points in train_points:
            if points.shape[0] >= target_num_points:
                # 随机采样
                indices = np.random.choice(points.shape[0], target_num_points, replace=False)
                train_points_padded.append(points[indices])
            else:
                # 重复采样
                indices = np.random.choice(points.shape[0], target_num_points, replace=True)
                train_points_padded.append(points[indices])

        for points in test_points:
            if points.shape[0] >= target_num_points:
                indices = np.random.choice(points.shape[0], target_num_points, replace=False)
                test_points_padded.append(points[indices])
            else:
                indices = np.random.choice(points.shape[0], target_num_points, replace=True)
                test_points_padded.append(points[indices])

        # 转换为numpy数组
        train_points_array = np.array(train_points_padded, dtype=np.float32)
        train_labels_array = train_labels.reshape(-1, 1).astype(np.int32)
        test_points_array = np.array(test_points_padded, dtype=np.float32)
        test_labels_array = test_labels.reshape(-1, 1).astype(np.int32)

        # 保存为HDF5
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('train_points', data=train_points_array)
            f.create_dataset('train_labels', data=train_labels_array)
            f.create_dataset('test_points', data=test_points_array)
            f.create_dataset('test_labels', data=test_labels_array)

            # 保存类别信息
            f.attrs['class_names'] = json.dumps(self.class_names)
            f.attrs['num_classes'] = len(self.class_names)
            f.attrs['num_train_samples'] = len(train_points_array)
            f.attrs['num_test_samples'] = len(test_points_array)

        print(f"HDF5数据集保存到: {h5_file}")
        print(f"  - 训练样本: {len(train_points_array)}")
        print(f"  - 测试样本: {len(test_points_array)}")
        print(f"  - 点数: {target_num_points}")

    def _load_split_data(self) -> None:
        """
        加载指定分割的数据
        """
        # 检查预处理数据是否存在
        split_file = os.path.join(self.processed_dir, f"{self.split}_data.npz")

        if not os.path.exists(split_file):
            print(f"预处理数据不存在: {split_file}")
            print("开始预处理数据集...")
            self.preprocess()

        # 加载数据
        try:
            data = np.load(split_file, allow_pickle=True)
            points = data['points']
            labels = data['labels'].flatten()  # 确保是一维数组
        except Exception as e:
            print(f"加载数据失败: {e}")
            print("尝试从HDF5文件加载...")
            self._load_from_h5()
            return

        # 转换为列表格式
        self.points = [points[i].astype(np.float32) for i in range(len(points))]
        self.labels = labels.tolist()

        print(f"加载 {self.split} 分割: {len(self.points)} 个样本")

    def _load_from_h5(self) -> None:
        """
        从HDF5文件加载数据
        """
        h5_file = os.path.join(self.processed_dir, "stanford3d_dataset.h5")

        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5文件不存在: {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            if self.split == "train":
                points = f['train_points'][:]
                labels = f['train_labels'][:].flatten()
            else:
                points = f['test_points'][:]
                labels = f['test_labels'][:].flatten()

        self.points = [points[i].astype(np.float32) for i in range(len(points))]
        self.labels = labels.tolist()

    def get_area_split_indices(self, train_areas: List[int] = [1, 2, 3, 4, 5],
                               test_areas: List[int] = [6]) -> Dict[str, List[int]]:
        """
        按区域分割数据集（更真实的评估）

        Args:
            train_areas: 训练区域列表
            test_areas: 测试区域列表

        Returns:
            Dict[str, List[int]]: 分割索引
        """
        # 需要从原始数据重新加载以获取区域信息
        # 这里简化实现，实际应该保存区域信息
        print("按区域分割需要重新处理数据，使用随机分割替代")
        return self.split_dataset()

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


def create_stanford3d_dataloader(config: DatasetConfig, split: str = "train",
                                 shuffle: bool = True) -> "torch.utils.data.DataLoader":
    """
    创建Stanford3D数据加载器

    Args:
        config: 数据集配置
        split: 数据分割
        shuffle: 是否打乱数据

    Returns:
        DataLoader: PyTorch数据加载器
    """
    import torch

    dataset = Stanford3DDataset(config, split)

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


def test_stanford3d():
    """测试Stanford3D数据集"""
    from config import DatasetConfig, DatasetType

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.STANFORD3D,
        data_dir="./data/stanford3d_test",
        num_points=1024,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print("测试Stanford3D数据集...")

    try:
        # 创建数据集
        dataset = Stanford3DDataset(config, split="train")
        print(f"数据集大小: {len(dataset)}")
        print(f"类别数量: {len(dataset.class_names)}")
        print(f"类别: {dataset.class_names}")

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
        dataloader = create_stanford3d_dataloader(config, split="train", shuffle=True)
        batch = next(iter(dataloader))
        print(f"批次点云形状: {batch['points'].shape}")
        print(f"批次标签形状: {batch['label'].shape}")

        print("Stanford3D数据集测试通过!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("注意: 测试需要Stanford3D数据集，请确保数据集已下载")


if __name__ == "__main__":
    test_stanford3d()