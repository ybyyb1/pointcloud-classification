#!/usr/bin/env python3
"""
Kaggle HDF5直接训练脚本 - 修复版
解决CUDA兼容性问题，添加CPU回退
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

class H5PointCloudDataset(Dataset):
    """直接从HDF5文件加载点云数据的数据集"""

    def __init__(self, h5_file, split='train', transform=None):
        """
        初始化数据集

        Args:
            h5_file: HDF5文件路径
            split: 数据分割 ('train' 或 'test')
            transform: 数据增强变换
        """
        self.h5_file = h5_file
        self.split = split
        self.transform = transform

        # 加载数据到内存
        with h5py.File(h5_file, 'r') as f:
            if split == 'train':
                self.points = f['train_points'][:]
                self.labels = f['train_labels'][:].flatten()
            else:  # test
                self.points = f['test_points'][:]
                self.labels = f['test_labels'][:].flatten()

            # 获取类别信息
            if 'class_names' in f.attrs:
                self.class_names = json.loads(f.attrs['class_names'])
            else:
                # 默认Stanford3D类别
                self.class_names = [
                    "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
                    "door", "floor", "sofa", "stairs", "table", "wall", "window"
                ]

            self.num_classes = len(self.class_names)

        print(f"加载 {split} 数据: {len(self.points)} 个样本, {self.num_classes} 个类别")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].astype(np.float32)
        label = self.labels[idx]

        # 转换为PyTorch张量
        points = torch.from_numpy(points).float()
        label = torch.tensor(label, dtype=torch.long)

        # 应用变换（如果有）
        if self.transform:
            points = self.transform(points)

        return points, label


class SimplePointTransformerCPU(nn.Module):
    """简化版Point Transformer模型 - CPU优化版"""

    def __init__(self, num_classes=14, num_points=1024, dim=384):
        super().__init__()

        # 输入转换 - 使用更简单的结构
        self.input_conv = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 简化Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x形状: (batch_size, num_points, 3)
        batch_size = x.shape[0]

        # 转换为 (batch_size, 3, num_points)
        x = x.transpose(1, 2)

        # 特征提取
        x = self.input_conv(x)  # (batch_size, 64, num_points)

        # 转换为 (batch_size, num_points, 64)
        x = x.transpose(1, 2)

        # Transformer
        x = self.transformer(x)  # (batch_size, num_points, 64)

        # 全局平均池化
        x = x.mean(dim=1)  # (batch_size, 64)

        # 分类
        x = self.classifier(x)  # (batch_size, num_classes)

        return x


def setup_device(force_cpu=False):
    """设置计算设备，处理CUDA兼容性问题"""
    if force_cpu:
        print("⚠️ 强制使用CPU模式")
        return torch.device('cpu')

    if torch.cuda.is_available():
        try:
            # 测试CUDA是否真的可用
            test_tensor = torch.tensor([1.0]).cuda()
            test_result = test_tensor * 2
            device_name = torch.cuda.get_device_name(0)
            cuda_capability = torch.cuda.get_device_capability(0)
            print(f"✅ CUDA可用: {device_name}")
            print(f"   计算能力: {cuda_capability[0]}.{cuda_capability[1]}")
            print(f"   内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return torch.device('cuda')
        except Exception as e:
            print(f"⚠️ CUDA检测到但无法使用: {e}")
            print("   将回退到CPU模式")
            return torch.device('cpu')
    else:
        print("⚠️ CUDA不可用，使用CPU模式")
        return torch.device('cpu')


def train_model(h5_file, experiment_name="h5_direct_train",
                epochs=30, batch_size=16, learning_rate=0.001,
                checkpoint_dir="/kaggle/working/checkpoints",
                force_cpu=False):
    """训练模型"""

    print("=" * 60)
    print("开始HDF5直接训练 - 修复版")
    print("=" * 60)

    # 设置设备（处理CUDA兼容性）
    device = setup_device(force_cpu)

    # 创建数据集
    print(f"加载数据: {h5_file}")
    train_dataset = H5PointCloudDataset(h5_file, split='train')
    test_dataset = H5PointCloudDataset(h5_file, split='test')

    # 调整批次大小以适应CPU
    if device.type == 'cpu':
        batch_size = min(batch_size, 8)  # CPU上使用更小的批次大小
        print(f"CPU模式: 调整批次大小为 {batch_size}")

    # 创建数据加载器
    num_workers = 0 if device.type == 'cpu' else 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    # 创建模型（使用CPU优化版）
    num_classes = train_dataset.num_classes
    model = SimplePointTransformerCPU(num_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 创建检查点目录
    experiment_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 训练循环
    print(f"\n开始训练，共 {epochs} 个epoch")
    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"批次大小: {batch_size}, 学习率: {learning_rate}")
    print(f"设备: {device}")

    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total

        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for points, labels in test_loader:
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * test_correct / test_total

        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
        print(f"  测试准确率: {test_accuracy:.2f}%")

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint_path = os.path.join(experiment_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'best_accuracy': best_accuracy,
                'device': str(device),
            }, checkpoint_path)
            print(f"  ✅ 保存最佳模型: {checkpoint_path} (准确率: {best_accuracy:.2f}%)")

        # 学习率调整
        scheduler.step()

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, checkpoint_path)

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    print(f"实验目录: {experiment_dir}")
    print("=" * 60)

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_accuracy': best_accuracy,
        'num_classes': num_classes,
        'class_names': train_dataset.class_names,
        'device': str(device),
        'force_cpu': force_cpu
    }

    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"训练历史保存到: {history_path}")

    return {
        'experiment_dir': experiment_dir,
        'best_accuracy': best_accuracy,
        'test_accuracy': test_accuracies[-1] if test_accuracies else 0.0,
        'device': str(device)
    }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Kaggle HDF5直接训练 - 修复版")
    parser.add_argument("--h5_file", type=str,
                       default="/kaggle/working/data/stanford3d/stanford3d_dataset.h5",
                       help="HDF5文件路径")
    parser.add_argument("--experiment", type=str, default="h5_direct_train",
                       help="实验名称")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--install_pytorch", action="store_true",
                       help="安装兼容的PyTorch版本")

    args = parser.parse_args()

    print("Kaggle HDF5直接训练脚本 - 修复版")
    print("=" * 60)
    print(f"HDF5文件: {args.h5_file}")
    print(f"实验名称: {args.experiment}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"强制CPU: {args.cpu}")
    print("=" * 60)

    # 如果需要，安装兼容的PyTorch版本
    if args.install_pytorch:
        print("安装兼容的PyTorch版本...")
        # Kaggle P100 GPU (sm_60) 兼容的PyTorch版本
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.2.0", "torchvision==0.17.0", "--index-url", "https://download.pytorch.org/whl/cu118"], check=False)
        print("✅ PyTorch重新安装完成")
        print("请重启kernel并重新运行脚本")
        return

    # 检查HDF5文件是否存在
    if not os.path.exists(args.h5_file):
        print(f"❌ 错误: HDF5文件不存在: {args.h5_file}")
        print("\n请确保:")
        print("1. 已下载Stanford3D预处理数据集到Kaggle")
        print("2. 数据集已附加到Notebook")
        print("3. 或者手动设置正确的HDF5文件路径")
        sys.exit(1)

    # 验证HDF5文件
    try:
        with h5py.File(args.h5_file, 'r') as f:
            print(f"✅ HDF5文件验证成功")
            print(f"训练样本: {f['train_points'].shape}")
            print(f"测试样本: {f['test_points'].shape}")
            if 'class_names' in f.attrs:
                class_names = json.loads(f.attrs['class_names'])
                print(f"类别数: {len(class_names)}")
    except Exception as e:
        print(f"❌ HDF5文件验证失败: {e}")
        sys.exit(1)

    # 显示PyTorch版本信息
    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except:
            print("无法获取GPU信息")

    # 训练模型
    try:
        results = train_model(
            h5_file=args.h5_file,
            experiment_name=args.experiment,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            force_cpu=args.cpu
        )

        print("\n🎉 训练成功完成!")
        print(f"实验目录: {results['experiment_dir']}")
        print(f"最佳准确率: {results['best_accuracy']:.2f}%")
        print(f"使用设备: {results['device']}")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("\n建议的解决方案:")
        print("1. 使用 --cpu 参数强制使用CPU模式:")
        print(f"   python {__file__} --cpu")
        print("2. 安装兼容的PyTorch版本:")
        print(f"   python {__file__} --install_pytorch")
        print("3. 检查Kaggle环境中的CUDA支持")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()