#!/usr/bin/env python3
"""
Kaggle云GPU训练启动脚本 - 简化版
解决NumPy初始化错误，直接开始训练
"""

import os
import sys
import subprocess
import warnings

def fix_numpy_issue():
    """修复NumPy初始化问题"""
    print("=" * 60)
    print("修复NumPy初始化问题")
    print("=" * 60)

    print("重新安装NumPy...")
    try:
        # 先卸载现有的NumPy
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"],
                      capture_output=True, text=True)

        # 安装兼容的NumPy版本
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3"],
                      check=True)
        print("✅ NumPy重新安装完成")
        return True
    except Exception as e:
        print(f"⚠️ NumPy修复失败: {e}")
        print("尝试使用现有NumPy...")
        return False

def install_kaggle_deps():
    """安装Kaggle专用依赖"""
    print("\n" + "=" * 60)
    print("安装Kaggle专用依赖")
    print("=" * 60)

    # 安装核心依赖（跳过已安装的）
    core_packages = [
        "torch==2.2.0",
        "torchvision==0.17.0",
        "h5py>=3.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "kaggle>=1.5.0",
    ]

    for package in core_packages:
        print(f"安装: {package}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"],
                          check=True, capture_output=True)
            print(f"  ✓ 完成")
        except Exception as e:
            print(f"  ⚠️ 安装失败: {e}")
            print(f"  尝试继续...")

    print("\n✅ 依赖安装完成")

def setup_environment():
    """设置Kaggle环境"""
    print("\n" + "=" * 60)
    print("设置Kaggle环境")
    print("=" * 60)

    # 检查是否在Kaggle环境中
    is_kaggle = os.path.exists('/kaggle') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

    if not is_kaggle:
        print("⚠️ 警告: 不在Kaggle环境中运行")
        return False

    print("✅ 检测到Kaggle环境")

    # 设置工作目录
    if not os.path.exists('/kaggle/working'):
        os.makedirs('/kaggle/working', exist_ok=True)

    os.chdir('/kaggle/working')
    print(f"当前目录: {os.getcwd()}")

    return True

def download_dataset():
    """下载Stanford3D数据集"""
    print("\n" + "=" * 60)
    print("下载Stanford3D数据集")
    print("=" * 60)

    # 检查是否已存在数据集
    dataset_path = '/kaggle/working/data/stanford3d'
    h5_file = os.path.join(dataset_path, 'stanford3d_dataset.h5')

    if os.path.exists(h5_file):
        print(f"✅ 数据集已存在: {h5_file}")
        return h5_file

    # 下载数据集
    print("下载Stanford3D预处理数据集...")
    try:
        # 使用Kaggle数据集
        subprocess.run(["kaggle", "datasets", "download", "-d", "sankalpsagar/stanford3ddataset"],
                      check=True)

        # 解压
        subprocess.run(["unzip", "stanford3ddataset.zip", "-d", "/kaggle/working/data/"],
                      check=True)

        print("✅ 数据集下载完成")

        # 查找HDF5文件
        import glob
        h5_files = glob.glob('/kaggle/working/data/**/*.h5', recursive=True)
        h5_files.extend(glob.glob('/kaggle/working/data/**/*.hdf5', recursive=True))

        if h5_files:
            return h5_files[0]
        else:
            print("⚠️ 未找到HDF5文件，使用默认路径")
            return '/kaggle/working/data/stanford3d_dataset.h5'

    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        print("使用备用数据集或手动下载")
        return None

def test_pytorch_gpu():
    """测试PyTorch GPU支持"""
    print("\n" + "=" * 60)
    print("测试PyTorch GPU支持")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"✅ GPU可用: {device_name}")
            print(f"   CUDA版本: {cuda_version}")
            print(f"   计算能力: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("⚠️ CUDA不可用，将使用CPU模式")
            return False
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def run_training(h5_file=None):
    """运行训练"""
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    # 如果没有提供HDF5文件，使用默认
    if not h5_file:
        h5_file = '/kaggle/working/data/stanford3d/stanford3d_dataset.h5'

    # 检查文件是否存在
    if not os.path.exists(h5_file):
        print(f"❌ HDF5文件不存在: {h5_file}")
        print("\n请手动下载数据集:")
        print("!kaggle datasets download -d sankalpsagar/stanford3ddataset")
        print("!unzip stanford3ddataset.zip -d /kaggle/working/data/")
        return False

    # 运行训练脚本
    cmd = [
        sys.executable, "scripts/kaggle_h5_direct_train_fixed.py",
        "--h5_file", h5_file,
        "--experiment", "kaggle_simple_training",
        "--epochs", "30",
        "--batch_size", "32",
        "--learning_rate", "0.001"
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print("✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")

        # 尝试CPU模式
        print("\n尝试CPU模式...")
        cmd.append("--cpu")
        try:
            subprocess.run(cmd, check=True)
            print("✅ CPU训练完成!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ CPU训练也失败: {e2}")
            return False

def main():
    """主函数"""
    print("Kaggle云GPU训练启动脚本 - 简化版")
    print("=" * 60)

    # 1. 修复NumPy问题
    fix_numpy_issue()

    # 2. 设置环境
    if not setup_environment():
        print("不在Kaggle环境，跳过特定设置")

    # 3. 安装依赖
    install_kaggle_deps()

    # 4. 测试GPU
    gpu_available = test_pytorch_gpu()

    if not gpu_available:
        print("\n⚠️ 注意: GPU不可用，训练将使用CPU模式")
        print("   这可能会显著增加训练时间")

    # 5. 下载数据集
    h5_file = download_dataset()

    # 6. 运行训练
    if h5_file:
        success = run_training(h5_file)
        if success:
            print("\n🎉 训练成功完成!")
            print(f"模型保存在: /kaggle/working/pointcloud-classification/checkpoints/")
        else:
            print("\n❌ 训练失败，请检查错误信息")
    else:
        print("\n❌ 无法获取数据集，训练中止")

    return True

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    success = main()
    sys.exit(0 if success else 1)