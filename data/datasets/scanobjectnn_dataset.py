"""
ScanObjectNN数据集加载器
"""

import os
import sys
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


def is_kaggle_environment() -> bool:
    """检查是否在Kaggle环境中运行"""
    return os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def find_kaggle_input_file(version: str) -> Optional[str]:
    """
    在Kaggle输入目录中查找数据集文件

    Args:
        version: 数据集版本（'main_split' 或 'pb_t50_rs_split'）

    Returns:
        Optional[str]: 找到的文件路径，如果未找到则返回None
    """
    if not is_kaggle_environment():
        return None

    # 可能的Kaggle输入目录
    possible_dirs = [
        '/kaggle/input/scanobjectnn',
        '/kaggle/input/scan-object-nn',
        '/kaggle/input/scanobjectnn-h5',
        '/kaggle/input/scanobjectnn-dataset',
    ]

    # 可能的文件名模式
    filename_patterns = [
        f"{version}.h5",
        f"{version}/training_objectdataset.h5",
        f"{version}/testing_objectdataset.h5",
        "training_objectdataset.h5",
        "testing_objectdataset.h5",
        "train.h5",
        "test.h5",
    ]

    for input_dir in possible_dirs:
        if not os.path.exists(input_dir):
            continue

        for pattern in filename_patterns:
            file_path = os.path.join(input_dir, pattern)
            if os.path.exists(file_path):
                print(f"在Kaggle输入目录中找到文件: {file_path}")
                return file_path

        # 如果没有匹配模式，递归搜索.h5文件
        import glob
        h5_files = glob.glob(os.path.join(input_dir, "**", "*.h5"), recursive=True)
        if h5_files:
            # 优先选择包含版本名称的文件
            for h5_file in h5_files:
                if version in h5_file:
                    print(f"在Kaggle输入目录中找到匹配版本的文件: {h5_file}")
                    return h5_file
            # 返回第一个找到的.h5文件
            print(f"在Kaggle输入目录中找到文件: {h5_files[0]}")
            return h5_files[0]

    return None


def validate_h5_file(filepath: str, min_size_kb: int = 10) -> bool:
    """
    验证h5文件是否有效

    Args:
        filepath: 文件路径
        min_size_kb: 最小文件大小（KB），避免空文件或错误页面

    Returns:
        bool: 文件是否有效
    """
    import os
    import h5py

    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return False

    # 检查文件大小
    file_size_kb = os.path.getsize(filepath) / 1024
    if file_size_kb < min_size_kb:
        print(f"文件太小 ({file_size_kb:.1f}KB)，可能无效: {filepath}")
        return False

    # 尝试打开文件
    try:
        with h5py.File(filepath, 'r') as f:
            # 检查必要的数据集是否存在（对于ScanObjectNN）
            required_datasets = ['train_points', 'train_labels', 'test_points', 'test_labels']
            for ds in required_datasets:
                if ds not in f:
                    print(f"文件缺少必需的数据集 '{ds}': {filepath}")
                    return False
            print(f"h5文件验证通过: {filepath} (大小: {file_size_kb:.1f}KB)")
            return True
    except Exception as e:
        print(f"h5文件无效: {filepath}, 错误: {e}")
        return False


def download_from_kaggle(dataset_name: str, version: str, output_dir: str) -> str:
    """
    从Kaggle数据集下载

    Args:
        dataset_name: Kaggle数据集名称（如 'hkustvgd/scanobjectnn'）
        version: 数据集版本（'main_split' 或 'pb_t50_rs_split'）
        output_dir: 输出目录

    Returns:
        str: 下载的文件路径
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
            # 继续尝试下载，公开数据集可能不需要认证

        print(f"从Kaggle数据集 {dataset_name} 下载 {version}...")

        # 下载数据集，添加force=True和quiet=True参数
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True, force=True, quiet=True)

        # 查找下载的文件
        h5_file = os.path.join(output_dir, f"{version}.h5")

        # 验证找到的文件
        def validate_and_return(file_path):
            if validate_h5_file(file_path):
                print(f"Kaggle下载成功且文件有效: {file_path}")
                return file_path
            else:
                print(f"文件无效，跳过: {file_path}")
                return None

        if os.path.exists(h5_file):
            valid_file = validate_and_return(h5_file)
            if valid_file:
                return valid_file

        # 尝试在解压后的目录中查找
        import glob
        h5_files = glob.glob(os.path.join(output_dir, "**", "*.h5"), recursive=True)
        # 优先选择包含版本名称的文件
        for file in h5_files:
            if version in file:
                valid_file = validate_and_return(file)
                if valid_file:
                    return valid_file

        # 如果没找到特定版本，验证所有文件
        for file in h5_files:
            valid_file = validate_and_return(file)
            if valid_file:
                return valid_file

        # 所有文件都无效
        raise FileNotFoundError(f"在Kaggle数据集中未找到有效的.h5文件")

    except Exception as e:
        print(f"Kaggle下载失败: {e}")
        print("请确保已安装kaggle API: pip install kaggle")
        print("并配置了Kaggle API密钥")
        print("或者手动下载数据集:")
        print(f"   !kaggle datasets download -d {dataset_name}")
        print(f"   !unzip {dataset_name.split('/')[-1]}.zip -d {output_dir}")
        raise


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
        import os  # 确保os在本地可用
        print(f"下载ScanObjectNN数据集 ({self.version})...")

        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)

        # 首先检查是否在Kaggle输入目录中已存在数据集文件
        kaggle_input_file = find_kaggle_input_file(self.version)
        if kaggle_input_file:
            print(f"使用Kaggle输入目录中的文件: {kaggle_input_file}")
            # 复制到目标位置
            import shutil
            shutil.copy2(kaggle_input_file, self.h5_file)
            print(f"文件已复制到: {self.h5_file}")
            return

        # 其次检查是否在Kaggle环境中，尝试通过Kaggle API下载
        if is_kaggle_environment():
            try:
                print("检测到Kaggle环境，尝试从Kaggle数据集下载...")
                kaggle_dataset = "hkustvgd/scanobjectnn"
                downloaded_file = download_from_kaggle(kaggle_dataset, self.version, self.data_dir)
                # 如果下载的文件路径与预期不同，将其移动到预期位置
                if downloaded_file != self.h5_file:
                    import shutil
                    shutil.copy2(downloaded_file, self.h5_file)
                    print(f"文件已复制到: {self.h5_file}")
                return
            except Exception as e:
                print(f"Kaggle下载失败，将尝试备用URL: {e}")

        # 备用方案：尝试多个可能的URL
        filename = f"{self.version}.h5"
        urls_to_try = []

        # 1. 使用配置中的URL
        if self.data_url:
            if self.data_url.endswith('.h5'):
                urls_to_try.append(self.data_url)
            else:
                # 假设是基础URL，添加文件名
                if self.data_url.endswith('/'):
                    urls_to_try.append(f"{self.data_url}{filename}")
                else:
                    urls_to_try.append(f"{self.data_url}/{filename}")

        # 修复：添加正确的GitHub URL格式
        # ScanObjectNN数据集的实际结构可能是：h5_files/main_split/training_objectdataset.h5 等
        # 尝试常见的文件命名模式
        possible_filenames = [
            f"{self.version}.h5",  # main_split.h5
            f"training_objectdataset.h5",  # 可能是实际的文件名
            f"testing_objectdataset.h5",
            f"train.h5",
            f"test.h5"
        ]

        # 2. 添加常见的GitHub URL
        base_urls = [
            "https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/",
            "https://raw.githubusercontent.com/hkust-vgd/scanobjectnn/master/h5_files/",
            # 其他可能的镜像或备份源
            "https://gitcode.net/mirrors/hkust-vgd/scanobjectnn/raw/master/h5_files/",
            "https://hub.fastgit.xyz/hkust-vgd/scanobjectnn/raw/master/h5_files/",
        ]
        for base_url in base_urls:
            urls_to_try.append(f"{base_url}{filename}")

        # 3. 尝试可能的直接文件URL（针对不同的文件名模式）
        for file_pattern in possible_filenames:
            urls_to_try.append(f"https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/{file_pattern}")
            urls_to_try.append(f"https://raw.githubusercontent.com/hkust-vgd/scanobjectnn/master/h5_files/{file_pattern}")

        # 4. 从环境变量读取额外的URL
        extra_urls = os.environ.get('SCANOBJECTNN_EXTRA_URLS', '')
        if extra_urls:
            for url in extra_urls.split(';'):
                url = url.strip()
                if url:
                    urls_to_try.append(url)

        # 去重
        urls_to_try = list(dict.fromkeys(urls_to_try))
        print(f"将尝试以下URL下载数据集（共{len(urls_to_try)}个）:")

        last_exception = None

        for url in urls_to_try:
            try:
                print(f"尝试从 {url} 下载...")
                urllib.request.urlretrieve(url, self.h5_file)
                print(f"下载完成: {self.h5_file}")
                # 验证下载的文件
                if validate_h5_file(self.h5_file):
                    print(f"文件验证成功: {self.h5_file}")
                    return
                else:
                    print(f"下载的文件无效，删除并尝试下一个URL...")
                    if os.path.exists(self.h5_file):
                        os.remove(self.h5_file)
                    continue
            except Exception as e:
                print(f"下载失败: {e}")
                last_exception = e
                continue

        # 所有URL都失败
        print(f"所有下载尝试都失败: {last_exception}")
        print("请手动下载ScanObjectNN数据集:")
        print(f"1. 访问: https://github.com/hkust-vgd/scanobjectnn")
        print(f"2. 下载文件: {filename}")
        print(f"3. 保存到: {self.h5_file}")
        print("或者使用Kaggle数据集:")
        print(f"   !kaggle datasets download -d hkustvgd/scanobjectnn")
        print(f"   !unzip scanobjectnn.zip -d {self.data_dir}")
        # 检查是否允许创建虚拟数据集
        allow_dummy = os.environ.get('SCANOBJECTNN_ALLOW_DUMMY', 'false').lower() == 'true'
        if allow_dummy:
            print("警告: 所有下载尝试失败，将创建虚拟数据集用于测试。")
            print("警告: 虚拟数据集仅用于功能测试，不能用于实际训练！")
            self._create_dummy_h5()
            print(f"虚拟数据集已创建: {self.h5_file}")
            return
        else:
            print("提示: 如需创建虚拟数据集进行测试，请设置环境变量 SCANOBJECTNN_ALLOW_DUMMY=true")
            raise last_exception

    def _create_dummy_h5(self) -> None:
        """
        创建虚拟数据集用于测试
        生成随机点云数据，模拟ScanObjectNN数据格式
        """
        print("创建虚拟ScanObjectNN数据集...")
        import h5py
        import numpy as np

        # 创建小的数据集：10个训练样本，5个测试样本，每个样本1024个点，15个类别
        n_train = 10
        n_test = 5
        n_points = 1024
        n_classes = len(self.class_names)

        # 生成随机点云（在单位球内）
        train_points = np.random.randn(n_train, n_points, 3).astype(np.float32)
        train_labels = np.random.randint(0, n_classes, size=(n_train, 1), dtype=np.int32)

        test_points = np.random.randn(n_test, n_points, 3).astype(np.float32)
        test_labels = np.random.randint(0, n_classes, size=(n_test, 1), dtype=np.int32)

        # 保存为h5文件
        with h5py.File(self.h5_file, 'w') as f:
            f.create_dataset('train_points', data=train_points)
            f.create_dataset('train_labels', data=train_labels)
            f.create_dataset('test_points', data=test_points)
            f.create_dataset('test_labels', data=test_labels)

        print(f"虚拟数据集已保存到: {self.h5_file}")
        print(f"训练样本: {n_train}, 测试样本: {n_test}, 点数: {n_points}, 类别: {n_classes}")

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