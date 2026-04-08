#!/usr/bin/env python3
"""
快速预处理Stanford3D数据集
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatasetConfig, DatasetType
from data.datasets.stanford3d_dataset import Stanford3DDataset

def main():
    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.STANFORD3D,
        data_dir="./data/stanford3d",  # 预处理后的数据将保存在这里
        num_points=1024,
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        # Stanford3D特定配置
        stanford3d_areas=[1, 2, 3, 4, 5, 6],
        stanford3d_classes_to_include=[],  # 空列表表示包含所有类别
    )

    print("开始预处理Stanford3D数据集...")
    print(f"原始数据目录: {os.path.abspath('./data/Stanford3dDataset_v1.2')}")
    print(f"处理后的数据目录: {os.path.abspath(config.data_dir)}")

    # 检查原始数据是否存在
    raw_data_dir = os.path.join(config.data_dir, "Stanford3dDataset_v1.2")
    if not os.path.exists(raw_data_dir):
        # 检查上级目录
        parent_raw = os.path.join(os.path.dirname(config.data_dir), "Stanford3dDataset_v1.2")
        if os.path.exists(parent_raw):
            print(f"找到原始数据在: {parent_raw}")
            # 创建符号链接或直接使用
            os.makedirs(config.data_dir, exist_ok=True)
            # 对于Windows，创建目录链接
            if not os.path.exists(raw_data_dir):
                import shutil
                # 尝试创建目录链接
                try:
                    os.symlink(parent_raw, raw_data_dir)
                    print(f"创建符号链接: {raw_data_dir} -> {parent_raw}")
                except:
                    # Windows可能需要管理员权限，直接复制
                    print("创建符号链接失败，将在预处理时直接从上级目录读取")
        else:
            print(f"错误: 未找到原始Stanford3D数据集")
            print(f"请确保 data/Stanford3dDataset_v1.2 目录存在")
            return 1

    # 创建数据集实例，这会自动触发预处理
    try:
        dataset = Stanford3DDataset(config, split="train")
        print("预处理完成!")
        print(f"数据集大小: {len(dataset)}")
        print(f"类别: {dataset.class_names}")

        # 检查处理后的文件
        processed_dir = os.path.join(config.data_dir, "processed")
        if os.path.exists(processed_dir):
            print(f"\n处理后的文件:")
            for f in os.listdir(processed_dir):
                file_path = os.path.join(processed_dir, f)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  {f}: {size_mb:.2f} MB")

        # 显示统计信息
        stats = dataset.get_statistics()
        print(f"\n数据集统计:")
        for key, value in stats.items():
            if isinstance(value, (list, dict)):
                if len(str(value)) > 100:
                    print(f"  {key}: {type(value).__name__} ({len(value)} 项)")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())