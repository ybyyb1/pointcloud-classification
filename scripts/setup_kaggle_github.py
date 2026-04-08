#!/usr/bin/env python3
"""
Kaggle GitHub集成脚本
用于在Kaggle环境中克隆GitHub仓库并设置环境
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_kaggle_environment():
    """检查是否在Kaggle环境中"""
    kaggle_paths = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/lib"
    ]

    for path in kaggle_paths:
        if os.path.exists(path):
            return True

    return False


def clone_github_repo(repo_url, target_dir=None):
    """克隆GitHub仓库"""
    if target_dir is None:
        target_dir = "/kaggle/working"

    print(f"克隆GitHub仓库: {repo_url}")
    print(f"目标目录: {target_dir}")

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 获取仓库名称
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(target_dir, repo_name)

    # 如果仓库已存在，更新它
    if os.path.exists(repo_path):
        print(f"仓库已存在: {repo_path}")
        print("更新仓库...")
        try:
            subprocess.run(["git", "-C", repo_path, "pull"], check=True)
            print("仓库更新成功")
        except subprocess.CalledProcessError as e:
            print(f"更新仓库失败: {e}")
            print("尝试重新克隆...")
            subprocess.run(["rm", "-rf", repo_path], check=False)
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
    else:
        # 克隆仓库
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)

    print(f"仓库克隆/更新完成: {repo_path}")
    return repo_path


def install_dependencies(requirements_file="requirements.txt"):
    """安装依赖包"""
    print(f"安装依赖包: {requirements_file}")

    if not os.path.exists(requirements_file):
        print(f"警告: 依赖文件不存在: {requirements_file}")
        return False

    try:
        # 安装依赖
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
        print("依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        return False


def setup_kaggle_directories():
    """设置Kaggle目录结构"""
    print("设置Kaggle目录结构...")

    directories = [
        "/kaggle/working/data",
        "/kaggle/working/checkpoints",
        "/kaggle/working/logs",
        "/kaggle/working/outputs",
        "/kaggle/working/visualizations",
        "/kaggle/working/submissions"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

    return directories


def configure_kaggle_gpu():
    """配置Kaggle GPU设置"""
    print("配置GPU设置...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"GPU数量: {gpu_count}")
            print(f"GPU型号: {device_name}")
            print(f"GPU内存: {gpu_memory:.2f} GB")

            # Kaggle P100有16GB显存，设置适当的限制
            if gpu_memory >= 16:
                # 保留一些内存给系统
                memory_fraction = 14.0 / gpu_memory
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                print(f"GPU内存限制设置为: {memory_fraction:.2f}")

            # 启用cuDNN自动优化
            torch.backends.cudnn.benchmark = True
            print("cuDNN自动优化已启用")
        else:
            print("警告: 没有可用的GPU，将使用CPU")

    except ImportError:
        print("警告: PyTorch未安装，跳过GPU配置")
    except Exception as e:
        print(f"GPU配置出错: {e}")


def create_kaggle_config(repo_path):
    """创建Kaggle配置文件"""
    print("创建Kaggle配置文件...")

    config = {
        "kaggle": {
            "enabled": True,
            "working_dir": "/kaggle/working",
            "input_dir": "/kaggle/input",
            "gpu_available": True,
            "memory_limit_gb": 16,
            "time_limit_hours": 9
        },
        "project": {
            "name": "pointcloud-classification",
            "repo_url": "https://github.com/ybyyb1/pointcloud-classification.git",
            "repo_path": repo_path
        },
        "training": {
            "use_amp": True,
            "batch_size": 16,
            "checkpoint_dir": "/kaggle/working/checkpoints",
            "experiment_name": "kaggle_exp_001"
        },
        "data": {
            "cache_datasets": True,
            "preprocessed_dir": "/kaggle/working/data/preprocessed"
        }
    }

    config_path = "/kaggle/working/kaggle_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Kaggle配置文件已创建: {config_path}")
    return config_path


def setup_environment_variables():
    """设置环境变量优化性能"""
    print("设置环境变量...")

    env_vars = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "PYTHONPATH": "/kaggle/working/pointcloud-classification:/kaggle/working"
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置环境变量: {key}={value}")

    return env_vars


def main():
    """主函数"""
    print("=" * 60)
    print("Kaggle GitHub集成设置")
    print("=" * 60)

    # 检查是否在Kaggle环境中
    if not check_kaggle_environment():
        print("警告: 不在Kaggle环境中，某些功能可能受限")
        print("继续设置本地环境...")

    # GitHub仓库URL
    github_repo = "https://github.com/ybyyb1/pointcloud-classification.git"

    # 克隆GitHub仓库
    repo_path = clone_github_repo(github_repo)

    # 切换到仓库目录
    os.chdir(repo_path)
    print(f"切换到目录: {repo_path}")

    # 安装依赖
    requirements_file = os.path.join(repo_path, "requirements.txt")
    install_dependencies(requirements_file)

    # 设置Kaggle目录结构
    setup_kaggle_directories()

    # 配置GPU
    configure_kaggle_gpu()

    # 创建Kaggle配置文件
    create_kaggle_config(repo_path)

    # 设置环境变量
    setup_environment_variables()

    # 验证设置
    print("\n" + "=" * 60)
    print("设置完成!")
    print("=" * 60)

    print("\n下一步:")
    print("1. 运行训练: python main.py train --kaggle --experiment_name kaggle_exp_001")
    print("2. 或运行: python scripts/train.py --kaggle --experiment kaggle_exp_001")
    print("3. 查看日志: cat /kaggle/working/logs/training.log")
    print("\nKaggle环境已准备就绪!")

    # 返回仓库路径供后续使用
    return repo_path


if __name__ == "__main__":
    try:
        repo_path = main()
        print(f"\n仓库路径: {repo_path}")
        print("您可以在Kaggle Notebook中导入项目模块:")
        print(f"import sys")
        print(f"sys.path.insert(0, '{repo_path}')")
        print("from main import download_scanobjectnn, train_model")
    except Exception as e:
        print(f"设置过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)