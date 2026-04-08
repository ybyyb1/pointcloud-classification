#!/usr/bin/env python3
"""
S3DIS数据集准备工具
提供多种方式准备S3DIS分类数据集
"""
import os
import sys
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DatasetConfig, DatasetType

def check_s3dis_exists(data_dir: str) -> bool:
    """
    检查S3DIS数据集是否已经存在

    Args:
        data_dir: 数据目录

    Returns:
        bool: 数据集是否存在
    """
    raw_dir = os.path.join(data_dir, "Stanford3dDataset_v1.2_Aligned_Version")
    processed_dir = os.path.join(data_dir, "processed")

    # 检查原始数据
    if os.path.exists(raw_dir):
        # 检查是否包含至少一个Area
        area_dirs = [d for d in os.listdir(raw_dir) if d.startswith("Area_")]
        if len(area_dirs) > 0:
            print(f"原始S3DIS数据已存在: {len(area_dirs)} 个区域")
            return True

    # 检查预处理数据
    if os.path.exists(processed_dir):
        # 检查是否有处理过的文件
        npz_files = [f for f in os.listdir(processed_dir) if f.endswith('.npz')]
        if len(npz_files) >= 3:  # train, val, test
            print(f"预处理S3DIS数据已存在: {len(npz_files)} 个文件")
            return True

    return False

def download_full_s3dis(data_dir: str, areas: List[int] = None) -> bool:
    """
    下载完整S3DIS数据集

    Args:
        data_dir: 数据目录
        areas: 区域列表

    Returns:
        bool: 是否成功
    """
    from data.datasets.s3dis_dataset import S3DISDataset
    from config import DatasetConfig, DatasetType

    print("=" * 80)
    print("下载完整S3DIS数据集")
    print("警告: S3DIS数据集约70GB，下载和处理需要较长时间")
    print("=" * 80)

    # 确认是否继续
    response = input("是否继续？(y/n): ").strip().lower()
    if response != 'y':
        print("取消下载")
        return False

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.S3DIS,
        data_dir=data_dir,
        num_points=1024,
        batch_size=32,
        s3dis_area=areas if areas else [1, 2, 3, 4, 5, 6],
        s3dis_classes_to_include=["table", "chair", "sofa", "bookcase", "board"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    try:
        # 创建数据集（会自动下载和处理）
        dataset = S3DISDataset(config, split="train")
        print(f"[OK] S3DIS数据集下载和处理完成")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   类别数量: {len(dataset.class_names)}")

        # 获取统计信息
        stats = dataset.get_statistics()
        print(f"   类别分布: {stats.get('class_distribution', {})}")

        return True

    except Exception as e:
        print(f"[X] S3DIS数据集处理失败: {e}")
        return False

def create_small_test_dataset(data_dir: str) -> bool:
    """
    创建小型测试数据集（用于快速测试）

    Args:
        data_dir: 数据目录

    Returns:
        bool: 是否成功
    """
    print("=" * 80)
    print("创建小型S3DIS测试数据集")
    print("生成合成数据用于快速测试")
    print("=" * 80)

    # 创建目录
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # ScanObjectNN类别
    classes = ["table", "chair", "sofa", "bookcase", "board"]
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    id_to_class = {i: cls for i, cls in enumerate(classes)}

    # 生成合成数据
    n_samples = 100
    n_points = 1024
    n_classes = len(classes)

    print(f"生成 {n_samples} 个样本，{n_points} 个点/样本，{n_classes} 个类别")

    # 为每个类别生成不同的点云模式（确保每个点云恰好有n_points个点）
    def generate_points_for_class(class_id: int, n_samples_per_class: int):
        points_list = []
        for i in range(n_samples_per_class):
            # 生成基础随机点云
            points = np.random.randn(n_points, 3) * 0.2

            # 根据类别应用不同的变换
            if class_id == 0:  # table: 扁平，集中在z=0附近
                points[:, 2] = points[:, 2] * 0.1  # 压扁高度
                points[:, 0:2] *= 1.5  # 更宽
            elif class_id == 1:  # chair: 有些点较高（靠背）
                # 随机选择一些点作为靠背
                back_indices = np.random.choice(n_points, size=n_points//3, replace=False)
                points[back_indices, 2] += np.random.uniform(0.5, 1.0)
                points[back_indices, 0] += np.random.uniform(0.2, 0.5)
            elif class_id == 2:  # sofa: 长形
                points[:, 0] *= 2.0  # 更长
                points[:, 2] = points[:, 2] * 0.15 + 0.3  # 抬升
            elif class_id == 3:  # bookcase: 分层
                # 创建分层效果
                for z in np.linspace(0, 1.5, 5):
                    layer_indices = np.random.choice(n_points, size=n_points//5, replace=False)
                    points[layer_indices, 2] = z + np.random.normal(0, 0.05, size=n_points//5)
                points[:, 0] *= 0.8
                points[:, 1] *= 0.4
            else:  # board: 平面，宽而薄
                points[:, 0] *= 1.5  # 更宽
                points[:, 1] *= 0.2  # 更薄
                points[:, 2] = np.abs(points[:, 2]) * 0.5

            # 添加随机旋转和缩放
            angle = np.random.uniform(0, 360)
            rad = np.radians(angle)
            rotation = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ])
            points = points @ rotation
            points *= np.random.uniform(0.8, 1.2)

            points_list.append(points.astype(np.float32))

        return points_list

    # 生成所有数据
    all_points = []
    all_labels = []

    samples_per_class = n_samples // n_classes
    for class_id in range(n_classes):
        class_points = generate_points_for_class(class_id, samples_per_class)
        all_points.extend(class_points)
        all_labels.extend([class_id] * len(class_points))

    # 保持为列表，稍后堆叠
    all_points_list = all_points  # 列表，每个元素形状为(1024, 3)
    all_labels_array = np.array(all_labels, dtype=np.int32)

    # 分割数据集
    n_samples = len(all_points_list)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_end = int(n_samples * 0.7)
    val_end = train_end + int(n_samples * 0.15)

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
        split_points_list = [all_points_list[i] for i in split_indices]
        split_labels = all_labels_array[split_indices]

        # 将点云列表堆叠为数组 (n_samples, 1024, 3)
        split_points_array = np.stack(split_points_list, axis=0)

        # 保存为npz文件
        save_path = os.path.join(processed_dir, f"{split_name}_data.npz")
        np.savez_compressed(save_path,
                          points=split_points_array,
                          labels=split_labels)

        print(f"  {split_name}分割: {len(split_points_array)} 个样本")
        print(f"    类别分布: {np.bincount(split_labels.flatten()).tolist()}")

    # 保存元数据
    metadata = {
        "dataset_type": "s3dis_synthetic",
        "total_samples": n_samples,
        "classes": classes,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "points_per_sample": n_points,
        "description": "S3DIS合成测试数据集，用于快速测试",
        "note": "这不是真实的S3DIS数据，仅用于功能测试",
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "split_sizes": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        }
    }

    metadata_file = os.path.join(processed_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] 小型测试数据集创建完成")
    print(f"   保存到: {processed_dir}")
    print(f"   元数据: {metadata_file}")

    return True

def prepare_pretrained_s3dis(data_dir: str) -> bool:
    """
    准备预训练的S3DIS数据集（从预处理的检查点）

    Args:
        data_dir: 数据目录

    Returns:
        bool: 是否成功
    """
    print("=" * 80)
    print("准备预训练的S3DIS数据集")
    print("从预处理的检查点加载")
    print("=" * 80)

    # 检查是否有预处理的检查点
    # 这里可以添加从云存储下载预处理数据的逻辑
    # 目前暂时不支持

    print("注意: 预训练的S3DIS数据集功能尚未实现")
    print("请选择其他选项:")
    print("1. 下载完整S3DIS数据集")
    print("2. 创建小型测试数据集")
    print("3. 跳过S3DIS，只使用ScanObjectNN")

    return False

def validate_s3dis_dataset(data_dir: str) -> Dict[str, Any]:
    """
    验证S3DIS数据集

    Args:
        data_dir: 数据目录

    Returns:
        Dict[str, Any]: 验证结果
    """
    processed_dir = os.path.join(data_dir, "processed")
    result = {
        "exists": False,
        "type": None,
        "samples": 0,
        "classes": 0,
        "splits": {}
    }

    if not os.path.exists(processed_dir):
        return result

    # 检查分割文件
    splits = ["train", "val", "test"]
    for split in splits:
        split_file = os.path.join(processed_dir, f"{split}_data.npz")
        if os.path.exists(split_file):
            try:
                data = np.load(split_file, allow_pickle=True)
                points = data['points']
                labels = data['labels']
                result["splits"][split] = {
                    "samples": len(points),
                    "classes": len(np.unique(labels))
                }
                result["exists"] = True
            except Exception as e:
                print(f"验证{split}分割失败: {e}")

    # 检查元数据
    metadata_file = os.path.join(processed_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            result["type"] = metadata.get("dataset_type", "unknown")
            result["metadata"] = metadata
        except Exception as e:
            print(f"读取元数据失败: {e}")

    # 汇总统计
    if result["exists"]:
        total_samples = sum(split["samples"] for split in result["splits"].values())
        result["samples"] = total_samples
        if "metadata" in result:
            result["classes"] = len(result["metadata"].get("classes", []))

    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="S3DIS数据集准备工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查当前状态
  python prepare_s3dis_dataset.py --check

  # 创建小型测试数据集
  python prepare_s3dis_dataset.py --small

  # 下载完整S3DIS数据集（警告: 70GB）
  python prepare_s3dis_dataset.py --full

  # 准备预训练数据集
  python prepare_s3dis_dataset.py --pretrained

  # 指定数据目录
  python prepare_s3dis_dataset.py --small --data_dir ./my_data

  # 完整流程：检查->小型测试->验证
  python prepare_s3dis_dataset.py --check --small --validate
        """
    )

    parser.add_argument("--check", action="store_true",
                       help="检查S3DIS数据集状态")
    parser.add_argument("--small", action="store_true",
                       help="创建小型测试数据集")
    parser.add_argument("--full", action="store_true",
                       help="下载完整S3DIS数据集（70GB）")
    parser.add_argument("--pretrained", action="store_true",
                       help="准备预训练数据集")
    parser.add_argument("--validate", action="store_true",
                       help="验证数据集")
    parser.add_argument("--data_dir", type=str, default="./data/s3dis_classification",
                       help="数据目录")
    parser.add_argument("--areas", type=int, nargs="+",
                       help="S3DIS区域列表（仅用于完整下载）")

    args = parser.parse_args()

    # 如果没有指定任何操作，显示帮助
    if not any([args.check, args.small, args.full, args.pretrained, args.validate]):
        parser.print_help()
        return

    print("S3DIS数据集准备工具")
    print(f"数据目录: {args.data_dir}")
    print()

    # 检查数据集状态
    if args.check or args.validate:
        print("检查S3DIS数据集状态...")
        result = validate_s3dis_dataset(args.data_dir)

        if result["exists"]:
            print(f"[OK] S3DIS数据集已存在")
            print(f"   类型: {result['type']}")
            print(f"   总样本数: {result['samples']}")
            print(f"   类别数: {result['classes']}")
            for split, info in result["splits"].items():
                print(f"   {split}分割: {info['samples']} 样本, {info['classes']} 类别")
            if "metadata" in result and "description" in result["metadata"]:
                print(f"   描述: {result['metadata']['description']}")
        else:
            print("[X] S3DIS数据集不存在或无效")
        print()

    # 如果只需要检查，则退出
    if args.check and not any([args.small, args.full, args.pretrained]):
        return

    # 创建小型测试数据集
    if args.small:
        success = create_small_test_dataset(args.data_dir)
        if not success:
            print("[X] 创建小型测试数据集失败")
            sys.exit(1)

    # 下载完整数据集
    if args.full:
        success = download_full_s3dis(args.data_dir, args.areas)
        if not success:
            print("[X] 下载完整S3DIS数据集失败")
            sys.exit(1)

    # 准备预训练数据集
    if args.pretrained:
        success = prepare_pretrained_s3dis(args.data_dir)
        if not success:
            print("注意: 预训练数据集准备失败，尝试其他选项")

    # 最终验证
    if args.validate:
        print("\n最终验证...")
        result = validate_s3dis_dataset(args.data_dir)
        if result["exists"]:
            print(f"[OK] 数据集验证通过")
            print(f"   总样本数: {result['samples']}")
            print(f"   类别数: {result['classes']}")
        else:
            print("[X] 数据集验证失败")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("S3DIS数据集准备完成!")
    print(f"数据位置: {args.data_dir}")
    print("\n下一步:")
    print("1. 训练模型: python main.py train --model point_transformer --dataset s3dis")
    print("2. 可视化数据: python main.py visualize --dataset s3dis")
    print("3. 比较模型: python main.py compare --models point_transformer,pointnet,dgcnn")

if __name__ == "__main__":
    main()