#!/usr/bin/env python3
"""
测试数据集修复代码
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# 设置环境变量，允许虚拟数据集
os.environ['SCANOBJECTNN_ALLOW_DUMMY'] = 'true'

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config import DatasetConfig, DatasetType

def test_dummy_dataset():
    """测试虚拟数据集创建和加载"""
    print("测试虚拟数据集...")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix='scanobjectnn_test_')
    print(f"临时目录: {temp_dir}")

    try:
        # 创建配置
        config = DatasetConfig(
            dataset_type=DatasetType.SCANOBJECTNN,
            data_dir=temp_dir,
            num_points=1024,
            batch_size=4,
            scanobjectnn_version="main_split",
            scanobjectnn_url="https://example.com/dummy"  # 无效URL，确保触发虚拟数据集创建
        )

        # 导入数据集类
        from data.datasets.scanobjectnn_dataset import ScanObjectNNDataset

        # 创建数据集实例（应该触发虚拟数据集创建）
        print("创建数据集实例...")
        dataset = ScanObjectNNDataset(config, split="train")

        print(f"数据集大小: {len(dataset)}")
        print(f"类别数量: {len(dataset.class_names)}")

        # 获取一个样本
        sample = dataset[0]
        print(f"样本点云形状: {sample['points'].shape}")
        print(f"样本标签: {sample['label']}")

        # 检查数据加载器创建
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader
        dataloader = create_scanobjectnn_dataloader(config, split="train", shuffle=False)
        batch = next(iter(dataloader))
        print(f"批次点云形状: {batch['points'].shape}")
        print(f"批次标签形状: {batch['label'].shape}")

        print("虚拟数据集测试通过!")
        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"已清理临时目录: {temp_dir}")

def test_kaggle_input_finder():
    """测试Kaggle输入目录查找函数"""
    print("测试Kaggle输入目录查找...")
    from data.datasets.scanobjectnn_dataset import find_kaggle_input_file

    # 模拟非Kaggle环境
    result = find_kaggle_input_file("main_split")
    print(f"非Kaggle环境结果: {result}")

    # 模拟Kaggle环境（通过设置环境变量）
    os.environ['KAGGLE_KERNEL_RUN_TYPE'] = 'Interactive'
    result = find_kaggle_input_file("main_split")
    print(f"Kaggle环境结果（无文件）: {result}")
    del os.environ['KAGGLE_KERNEL_RUN_TYPE']

    print("Kaggle输入目录查找测试完成")

if __name__ == "__main__":
    print("=" * 60)
    print("数据集修复测试")
    print("=" * 60)

    # 测试Kaggle输入目录查找
    test_kaggle_input_finder()
    print()

    # 测试虚拟数据集
    success = test_dummy_dataset()
    print()

    if success:
        print("所有测试通过！")
        sys.exit(0)
    else:
        print("测试失败！")
        sys.exit(1)