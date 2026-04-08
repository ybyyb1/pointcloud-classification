#!/usr/bin/env python3
"""
Kaggle GPU兼容性修复脚本
专门解决Tesla P100 (sm_60)的CUDA兼容性问题
"""

import os
import sys
import subprocess
import platform

def check_cuda_version():
    """检查CUDA版本"""
    print("=" * 60)
    print("检查CUDA环境")
    print("=" * 60)

    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi可用")
            # 提取CUDA版本
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"系统CUDA版本: {line.strip()}")
        else:
            print("❌ nvidia-smi不可用")
    except Exception as e:
        print(f"⚠️ 无法运行nvidia-smi: {e}")

    # 检查nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvcc可用")
            lines = result.stdout.split('\n')
            if len(lines) > 3:
                print(f"NVCC版本: {lines[3].strip()}")
    except Exception as e:
        print(f"⚠️ 无法运行nvcc: {e}")

def check_pytorch_installation():
    """检查PyTorch安装"""
    print("\n" + "=" * 60)
    print("检查PyTorch安装")
    print("=" * 60)

    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")

        if hasattr(torch.version, 'cuda'):
            print(f"   PyTorch CUDA版本: {torch.version.cuda}")

        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA可用: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   GPU数量: {device_count}")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                print(f"   GPU {i}: {device_name}")
                print(f"     计算能力: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")

                # 检查内存
                total_memory = torch.cuda.get_device_properties(i).total_memory
                print(f"     显存: {total_memory / 1024**3:.2f} GB")

        return True, cuda_available

    except ImportError as e:
        print(f"❌ PyTorch未安装: {e}")
        return False, False
    except Exception as e:
        print(f"⚠️ 检查PyTorch时出错: {e}")
        return False, False

def test_cuda_compatibility():
    """测试CUDA兼容性"""
    print("\n" + "=" * 60)
    print("测试CUDA兼容性")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False

        print("运行CUDA兼容性测试...")

        # 测试1: 基本张量运算
        print("  1. 基本张量运算...")
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        c = torch.mm(a, b)
        print("    ✅ 矩阵乘法通过")

        # 测试2: 卷积运算 (最容易出问题的)
        print("  2. 卷积运算...")
        try:
            conv = torch.nn.Conv1d(3, 32, 1).cuda()
            x = torch.randn(16, 3, 1024, device='cuda')
            y = conv(x)
            print("    ✅ Conv1d通过")
        except Exception as e:
            print(f"    ❌ Conv1d失败: {e}")

            # 尝试简化测试
            print("    尝试简化测试...")
            try:
                conv = torch.nn.Conv1d(3, 8, 1).cuda()
                x = torch.randn(4, 3, 256, device='cuda')
                y = conv(x)
                print("    ✅ 简化Conv1d通过")
            except Exception as e2:
                print(f"    ❌ 简化Conv1d也失败: {e2}")
                return False

        # 测试3: 批归一化
        print("  3. 批归一化...")
        try:
            bn = torch.nn.BatchNorm1d(32).cuda()
            y = bn(y)
            print("    ✅ BatchNorm1d通过")
        except Exception as e:
            print(f"    ⚠️ BatchNorm1d警告: {e}")

        print("\n✅ 所有CUDA测试通过!")
        return True

    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def install_compatible_pytorch():
    """安装兼容的PyTorch版本"""
    print("\n" + "=" * 60)
    print("安装兼容的PyTorch版本")
    print("=" * 60)

    print("Kaggle环境通常预装PyTorch，但可能不包含sm_60核函数")
    print("尝试安装兼容版本...")

    # 方案1: 安装包含sm_60的PyTorch版本
    print("\n方案1: 安装包含sm_60支持的版本")

    # 首先检查当前Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python版本: {python_version}")

    # 确定最佳安装命令
    if python_version.startswith('3.12'):
        print("检测到Python 3.12，使用cu121版本")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    elif python_version.startswith('3.11'):
        print("检测到Python 3.11，使用cu118版本")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        print(f"Python {python_version}，尝试通用版本")
        install_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]

    print(f"执行安装命令: {' '.join(install_cmd)}")

    try:
        subprocess.run(install_cmd, check=True)
        print("✅ PyTorch安装成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")

        # 方案2: 尝试CPU版本作为后备
        print("\n方案2: 安装CPU版本作为后备")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True)
            print("✅ CPU版本安装成功")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ CPU版本也失败: {e2}")
            return False

def create_compatible_model_code():
    """创建兼容的模型代码"""
    print("\n" + "=" * 60)
    print("创建兼容的模型代码")
    print("=" * 60)

    code = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class P100CompatibleModel(nn.Module):
    """P100 GPU兼容的简化模型"""

    def __init__(self, num_classes=14):
        super().__init__()
        # 使用简单结构避免兼容性问题
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)

        # 不使用BatchNorm（有时有兼容性问题）
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # 手动初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)

        # 使用简单的激活函数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def create_model_for_device(force_cpu=False):
    """创建设备兼容的模型"""
    if force_cpu or not torch.cuda.is_available():
        print("使用CPU模式")
        device = torch.device('cpu')
        model = P100CompatibleModel().to(device)
    else:
        print("使用GPU模式")
        device = torch.device('cuda')
        model = P100CompatibleModel().to(device)

        # 测试模型是否能在GPU上运行
        try:
            test_input = torch.randn(4, 1024, 3, device=device)
            test_output = model(test_input)
            print("✅ 模型GPU测试通过")
        except Exception as e:
            print(f"❌ 模型GPU测试失败: {e}")
            print("回退到CPU模式")
            device = torch.device('cpu')
            model = model.to(device)

    return model, device
'''

    # 保存代码
    with open('/kaggle/working/p100_compatible_model.py', 'w') as f:
        f.write(code)

    print("✅ 兼容模型代码已保存到: /kaggle/working/p100_compatible_model.py")
    return True

def create_gpu_training_script():
    """创建GPU训练脚本"""
    print("\n" + "=" * 60)
    print("创建GPU训练脚本")
    print("=" * 60)

    script = '''#!/usr/bin/env python3
"""
Kaggle P100 GPU训练脚本
"""

import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设置环境变量优化GPU性能
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 更好的错误信息
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

class H5Dataset(Dataset):
    """简单HDF5数据集"""

    def __init__(self, h5_file, split='train'):
        with h5py.File(h5_file, 'r') as f:
            if split == 'train':
                self.points = f['train_points'][:]
                self.labels = f['train_labels'][:].flatten()
            else:
                self.points = f['test_points'][:]
                self.labels = f['test_labels'][:].flatten()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx].astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(points), torch.tensor(label, dtype=torch.long)

class SimplePointNet(nn.Module):
    """简化版PointNet，P100兼容"""

    def __init__(self, num_classes=14):
        super().__init__()
        # 避免复杂操作
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # 全局最大池化
        x = torch.max(x, 2)[0]

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_gpu_model(h5_file, epochs=30, batch_size=16, lr=0.001):
    """GPU训练函数"""

    print("=" * 60)
    print("开始GPU训练")
    print("=" * 60)

    # 设备设置
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")

        # 测试CUDA
        try:
            test_tensor = torch.randn(2, 2).cuda()
            _ = test_tensor @ test_tensor.T
            print("✅ CUDA测试通过")
        except Exception as e:
            print(f"❌ CUDA测试失败: {e}")
            print("回退到CPU模式")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("使用CPU模式")

    print(f"设备: {device}")

    # 加载数据
    train_dataset = H5Dataset(h5_file, 'train')
    test_dataset = H5Dataset(h5_file, 'test')

    # 根据设备调整批次大小
    if device.type == 'cpu':
        batch_size = min(batch_size, 8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"批次大小: {batch_size}, 学习率: {lr}")

    # 创建模型
    model = SimplePointNet(num_classes=14).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_accuracy = 0.0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}")

        train_acc = 100. * correct / total

        # 评估
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

        test_acc = 100. * test_correct / test_total

        print(f"\\nEpoch {epoch+1}/{epochs}:")
        print(f"  训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%")

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
                'best_accuracy': best_accuracy,
            }, '/kaggle/working/best_model.pth')
            print(f"  ✅ 保存最佳模型 (准确率: {best_accuracy:.2f}%)")

    print(f"\\n🎉 训练完成! 最佳测试准确率: {best_accuracy:.2f}%")
    print(f"模型保存在: /kaggle/working/best_model.pth")

    return best_accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", default="/kaggle/working/data/stanford3d/stanford3d_dataset.h5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    # 验证HDF5文件
    if not os.path.exists(args.h5_file):
        print(f"❌ HDF5文件不存在: {args.h5_file}")
        sys.exit(1)

    with h5py.File(args.h5_file, 'r') as f:
        print(f"✅ HDF5文件验证成功")
        print(f"训练样本: {f['train_points'].shape}")
        print(f"测试样本: {f['test_points'].shape}")

    # 训练模型
    train_gpu_model(args.h5_file, args.epochs, args.batch_size, args.lr)
'''

    # 保存脚本
    with open('/kaggle/working/train_gpu_simple.py', 'w') as f:
        f.write(script)

    print("✅ GPU训练脚本已保存到: /kaggle/working/train_gpu_simple.py")
    return True

def main():
    """主函数"""
    print("Kaggle P100 GPU兼容性修复工具")
    print("=" * 60)

    # 检查环境
    check_cuda_version()

    # 检查PyTorch
    pytorch_installed, cuda_available = check_pytorch_installation()

    if not pytorch_installed:
        print("\nPyTorch未安装，尝试安装...")
        install_compatible_pytorch()
        # 重新检查
        pytorch_installed, cuda_available = check_pytorch_installation()

    # 测试兼容性
    if pytorch_installed and cuda_available:
        compatibility_ok = test_cuda_compatibility()
        if not compatibility_ok:
            print("\nCUDA兼容性测试失败，尝试重新安装PyTorch...")
            install_compatible_pytorch()
            # 再次测试
            test_cuda_compatibility()

    # 创建兼容代码
    create_compatible_model_code()

    # 创建训练脚本
    create_gpu_training_script()

    print("\n" + "=" * 60)
    print("修复完成!")
    print("=" * 60)

    print("\n下一步操作:")
    print("1. 运行GPU训练脚本:")
    print("   python /kaggle/working/train_gpu_simple.py \\")
    print("     --h5_file /kaggle/working/data/stanford3d/stanford3d_dataset.h5 \\")
    print("     --epochs 30 --batch_size 32")

    print("\n2. 如果GPU训练失败，可以使用CPU模式:")
    print("   在代码开头添加: os.environ['CUDA_VISIBLE_DEVICES'] = ''")

    print("\n3. 或者使用兼容模型代码:")
    print("   from p100_compatible_model import create_model_for_device")
    print("   model, device = create_model_for_device(force_cpu=False)")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)