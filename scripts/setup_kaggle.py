#!/usr/bin/env python3
"""
Kaggle环境配置脚本
用于在Kaggle环境中设置点云分类系统
"""

import os
import sys
import json
import shutil
from pathlib import Path


def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("设置Kaggle环境...")

    # Kaggle特定路径
    kaggle_input_path = "/kaggle/input"
    kaggle_working_path = "/kaggle/working"

    # 检查是否在Kaggle环境中
    if not os.path.exists(kaggle_input_path):
        print("警告: 不在Kaggle环境中，某些功能可能受限")
        return False

    print(f"Kaggle输入目录: {kaggle_input_path}")
    print(f"Kaggle工作目录: {kaggle_working_path}")

    # 创建必要的目录结构
    directories = [
        os.path.join(kaggle_working_path, "data"),
        os.path.join(kaggle_working_path, "checkpoints"),
        os.path.join(kaggle_working_path, "logs"),
        os.path.join(kaggle_working_path, "outputs"),
        os.path.join(kaggle_working_path, "visualizations"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

    # 设置Kaggle API配置（如果存在）
    kaggle_config_path = "/kaggle/input/kaggle-api-config/kaggle.json"
    if os.path.exists(kaggle_config_path):
        # 复制Kaggle配置文件到正确位置
        kaggle_home = Path.home() / ".kaggle"
        kaggle_home.mkdir(exist_ok=True)
        shutil.copy(kaggle_config_path, kaggle_home / "kaggle.json")
        os.chmod(kaggle_home / "kaggle.json", 0o600)
        print("Kaggle API配置已设置")

    # 创建Kaggle特定的配置文件
    create_kaggle_config(kaggle_working_path)

    print("Kaggle环境设置完成!")
    return True


def create_kaggle_config(working_dir: str):
    """创建Kaggle特定配置文件"""
    config = {
        "kaggle": {
            "enabled": True,
            "working_dir": working_dir,
            "input_dir": "/kaggle/input",
            "gpu_available": True,
            "memory_limit_gb": 16,  # P100有16GB显存
            "time_limit_hours": 9,  # Kaggle会话时间限制
        },
        "training": {
            "use_amp": True,  # 自动混合精度训练
            "batch_size": 16,  # Kaggle上建议使用较小的批次大小
            "checkpoint_dir": os.path.join(working_dir, "checkpoints"),
            "save_checkpoint_interval": 5,
            "early_stopping_patience": 10,
        },
        "data": {
            "cache_datasets": True,  # 缓存数据集到工作目录
            "preprocessed_dir": os.path.join(working_dir, "data/preprocessed"),
        },
        "output": {
            "submission_dir": os.path.join(working_dir, "submissions"),
            "model_dir": os.path.join(working_dir, "models"),
            "log_dir": os.path.join(working_dir, "logs"),
        }
    }

    config_path = os.path.join(working_dir, "kaggle_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Kaggle配置文件已创建: {config_path}")
    return config_path


def download_kaggle_datasets():
    """下载Kaggle数据集"""
    print("下载Kaggle数据集...")

    try:
        import kaggle

        # ScanObjectNN数据集（如果在Kaggle上可用）
        datasets = [
            {
                "name": "scanobjectnn",
                "dataset": "hkustvgd/scanobjectnn",  # 示例，实际可能需要调整
                "description": "ScanObjectNN点云分类数据集"
            }
        ]

        download_dir = "/kaggle/working/data"
        os.makedirs(download_dir, exist_ok=True)

        for dataset_info in datasets:
            print(f"下载数据集: {dataset_info['name']}")
            try:
                kaggle.api.dataset_download_files(
                    dataset_info["dataset"],
                    path=os.path.join(download_dir, dataset_info["name"]),
                    unzip=True
                )
                print(f"数据集 {dataset_info['name']} 下载完成")
            except Exception as e:
                print(f"下载数据集 {dataset_info['name']} 失败: {e}")

    except ImportError:
        print("Kaggle API未安装，跳过数据集下载")
    except Exception as e:
        print(f"数据集下载过程出错: {e}")


def setup_gpu_config():
    """设置GPU配置"""
    print("设置GPU配置...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)

            print(f"GPU数量: {gpu_count}")
            print(f"当前GPU: {device_name}")
            print(f"GPU内存: {gpu_memory:.2f} GB")

            # 设置GPU内存限制（Kaggle P100有16GB）
            if gpu_memory >= 16:
                # 为Kaggle环境保留一些内存
                memory_fraction = 14.0 / gpu_memory  # 使用14GB
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                print(f"GPU内存限制设置为: {memory_fraction:.2f}")

            # 设置cuDNN自动优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("cuDNN自动优化已启用")

        else:
            print("警告: 没有可用的GPU，将使用CPU")

    except Exception as e:
        print(f"GPU配置出错: {e}")


def optimize_kaggle_performance():
    """优化Kaggle性能"""
    print("优化Kaggle性能...")

    # 设置环境变量
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    print("环境变量已设置以优化性能")

    # 创建性能优化配置
    config = {
        "performance": {
            "num_workers": 2,  # Kaggle上建议使用较少的worker
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "training": {
            "gradient_accumulation_steps": 2,  # 模拟更大批次
            "mixed_precision": True,
            "gradient_clipping": 1.0,
        }
    }

    return config


def create_kaggle_notebook():
    """创建Kaggle Notebook示例"""
    print("创建Kaggle Notebook示例...")

    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 点云分类系统 - Kaggle Notebook\n",
    "## 基于点云数据的室内场景三维物体分类\n",
    "\n",
    "这个Notebook演示了如何在Kaggle上使用点云分类系统。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 设置环境\n",
    "!pip install -r requirements.txt\n",
    "!python scripts/setup_kaggle.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 导入库\n",
    "import sys\n",
    "sys.path.append('/kaggle/working')\n",
    "\n",
    "from main import download_scanobjectnn, train_model, evaluate_model\n",
    "import argparse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 下载数据集\n",
    "args = argparse.Namespace(\n",
    "    data_dir='/kaggle/working/data/scanobjectnn',\n",
    "    version='main_split',\n",
    "    num_points=1024,\n",
    "    batch_size=16\n",
    ")\n",
    "\n",
    "download_scanobjectnn(args)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 训练模型\n",
    "args = argparse.Namespace(\n",
    "    model='point_transformer',\n",
    "    dataset='scanobjectnn',\n",
    "    data_dir='/kaggle/working/data/scanobjectnn',\n",
    "    epochs=50,\n",
    "    batch_size=16,\n",
    "    learning_rate=0.001,\n",
    "    early_stopping=True,\n",
    "    kaggle=True,\n",
    "    experiment_name='kaggle_exp_001'\n",
    ")\n",
    "\n",
    "train_model(args)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 评估模型\n",
    "args = argparse.Namespace(\n",
    "    checkpoint='/kaggle/working/checkpoints/kaggle_exp_001/best_model.pth',\n",
    "    dataset='scanobjectnn',\n",
    "    data_dir='/kaggle/working/data/scanobjectnn',\n",
    "    batch_size=16\n",
    ")\n",
    "\n",
    "evaluate_model(args)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    notebook_path = "/kaggle/working/pointcloud_classification_kaggle.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(notebook_content)

    print(f"Kaggle Notebook示例已创建: {notebook_path}")
    return notebook_path


def main():
    """主函数"""
    print("=" * 60)
    print("点云分类系统 - Kaggle环境配置")
    print("=" * 60)

    # 设置Kaggle环境
    if setup_kaggle_environment():
        # 设置GPU配置
        setup_gpu_config()

        # 优化性能
        optimize_kaggle_performance()

        # 下载数据集（可选）
        print("\n是否下载数据集? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            download_kaggle_datasets()

        # 创建Notebook示例
        print("\n是否创建Kaggle Notebook示例? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            create_kaggle_notebook()

        print("\n" + "=" * 60)
        print("配置完成!")
        print("\n下一步:")
        print("1. 运行训练: python main.py train --kaggle")
        print("2. 查看日志: cat /kaggle/working/logs/training.log")
        print("3. 提交结果到Kaggle")
        print("=" * 60)
    else:
        print("Kaggle环境设置失败")


if __name__ == "__main__":
    main()