"""
数据集配置文件
提供数据集相关的配置和工具函数
"""

from typing import List, Dict, Any, Optional
from .base_config import DatasetConfig, DatasetType


# 预定义的数据集配置
SCANOBJECTNN_CONFIG = DatasetConfig(
    dataset_type=DatasetType.SCANOBJECTNN,
    data_dir="./data/scanobjectnn",
    num_points=1024,
    batch_size=32,
    scanobjectnn_version="main_split",
    use_augmentation=True
)

S3DIS_CONFIG = DatasetConfig(
    dataset_type=DatasetType.S3DIS,
    data_dir="./data/s3dis",
    num_points=1024,
    batch_size=16,  # S3DIS数据更大，减小批量大小
    s3dis_area=[1, 2, 3, 4, 5, 6],
    s3dis_classes_to_include=[
        "table", "chair", "sofa", "bookcase", "board"
    ],
    use_augmentation=True
)

STANFORD3D_CONFIG = DatasetConfig(
    dataset_type=DatasetType.STANFORD3D,
    data_dir="./data/stanford3d",
    num_points=1024,
    batch_size=32,
    stanford3d_areas=[1, 2, 3, 4, 5, 6],
    stanford3d_classes_to_include=[
        "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
        "door", "floor", "sofa", "stairs", "table", "wall", "window"
    ],
    use_augmentation=True
)

# ScanObjectNN类别列表（15个类别）
SCANOBJECTNN_CLASSES = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
    "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet"
]

# ScanObjectNN类别到ID的映射
SCANOBJECTNN_CLASS_TO_ID = {cls: i for i, cls in enumerate(SCANOBJECTNN_CLASSES)}
SCANOBJECTNN_ID_TO_CLASS = {i: cls for i, cls in enumerate(SCANOBJECTNN_CLASSES)}

# Stanford3D类别列表（14个类别）
STANFORD3D_CLASSES = [
    "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
    "door", "floor", "sofa", "stairs", "table", "wall", "window"
]

# Stanford3D类别到ID的映射
STANFORD3D_CLASS_TO_ID = {cls: i for i, cls in enumerate(STANFORD3D_CLASSES)}
STANFORD3D_ID_TO_CLASS = {i: cls for i, cls in enumerate(STANFORD3D_CLASSES)}

# S3DIS到ScanObjectNN的类别映射
S3DIS_TO_SCANOBJECTNN_MAPPING = {
    "table": "table",
    "chair": "chair",
    "sofa": "sofa",
    "bookcase": "shelf",
    "board": "display",
    # 其他映射可以根据需要添加
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    根据数据集名称获取配置

    Args:
        dataset_name: 数据集名称，支持 "scanobjectnn", "s3dis", "stanford3d"

    Returns:
        DatasetConfig: 数据集配置对象
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "scanobjectnn":
        return SCANOBJECTNN_CONFIG
    elif dataset_name == "s3dis":
        return S3DIS_CONFIG
    elif dataset_name == "stanford3d":
        return STANFORD3D_CONFIG
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


def get_class_names(dataset_name: str) -> List[str]:
    """
    获取数据集的类别名称列表

    Args:
        dataset_name: 数据集名称

    Returns:
        List[str]: 类别名称列表
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "scanobjectnn":
        return SCANOBJECTNN_CLASSES
    elif dataset_name == "s3dis":
        # S3DIS使用映射后的类别
        return list(set(S3DIS_TO_SCANOBJECTNN_MAPPING.values()))
    elif dataset_name == "stanford3d":
        return STANFORD3D_CLASSES
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")


if __name__ == "__main__":
    # 测试函数
    config = get_dataset_config("scanobjectnn")
    print(f"ScanObjectNN配置: {config}")

    classes = get_class_names("scanobjectnn")
    print(f"ScanObjectNN类别: {classes}")

    config2 = get_dataset_config("s3dis")
    print(f"S3DIS配置: {config2}")