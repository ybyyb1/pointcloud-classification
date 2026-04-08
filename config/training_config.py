"""
训练配置文件
提供训练相关的配置和工具函数
"""

from typing import List, Dict, Any, Optional
from .base_config import TrainingConfig


# 预定义的训练配置
DEFAULT_TRAINING_CONFIG = TrainingConfig(
    epochs=300,
    learning_rate=0.001,
    weight_decay=0.01,
    optimizer="adamw",
    scheduler="cosine",
    use_amp=True,
    gradient_accumulation_steps=1,
    early_stopping_patience=20,
    checkpoint_dir="./checkpoints"
)

# Kaggle优化配置（针对P100 GPU）
KAGGLE_TRAINING_CONFIG = TrainingConfig(
    epochs=200,  # Kaggle会话时间有限，减少轮数
    learning_rate=0.001,
    batch_size=16,  # P100内存较小，减小批量大小
    use_amp=True,
    gradient_accumulation_steps=2,  # 模拟更大的批量大小
    checkpoint_dir="/kaggle/working/checkpoints",
    kaggle_output_dir="/kaggle/working",
    kaggle_gpu_memory_limit=14.0  # P100有16GB，保留2GB给系统
)

# 快速训练配置（用于调试）
FAST_TRAINING_CONFIG = TrainingConfig(
    epochs=10,
    learning_rate=0.001,
    batch_size=8,
    use_amp=False,  # 调试时关闭AMP
    save_checkpoint_interval=5,
    early_stopping_patience=5
)


def get_training_config(config_name: str = "default") -> TrainingConfig:
    """
    根据配置名称获取训练配置

    Args:
        config_name: 配置名称，支持 "default", "kaggle", "fast"

    Returns:
        TrainingConfig: 训练配置对象
    """
    config_name = config_name.lower()
    if config_name == "default":
        return DEFAULT_TRAINING_CONFIG
    elif config_name == "kaggle":
        return KAGGLE_TRAINING_CONFIG
    elif config_name == "fast":
        return FAST_TRAINING_CONFIG
    else:
        raise ValueError(f"未知的训练配置: {config_name}")


def create_optimizer_config(optimizer_name: str) -> Dict[str, Any]:
    """
    创建优化器配置

    Args:
        optimizer_name: 优化器名称

    Returns:
        Dict[str, Any]: 优化器配置字典
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return {
            "type": "adam",
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    elif optimizer_name == "adamw":
        return {
            "type": "adamw",
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    elif optimizer_name == "sgd":
        return {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "nesterov": True
        }
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")


def create_scheduler_config(scheduler_name: str, total_epochs: int) -> Dict[str, Any]:
    """
    创建学习率调度器配置

    Args:
        scheduler_name: 调度器名称
        total_epochs: 总训练轮数

    Returns:
        Dict[str, Any]: 调度器配置字典
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        return {
            "type": "cosine",
            "T_max": total_epochs,
            "eta_min": 1e-6
        }
    elif scheduler_name == "step":
        return {
            "type": "step",
            "milestones": [total_epochs // 3, total_epochs * 2 // 3],
            "gamma": 0.1
        }
    elif scheduler_name == "plateau":
        return {
            "type": "plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6
        }
    else:
        raise ValueError(f"未知的调度器: {scheduler_name}")


def estimate_training_time(config: TrainingConfig, dataset_size: int) -> str:
    """
    估计训练时间

    Args:
        config: 训练配置
        dataset_size: 数据集大小（样本数）

    Returns:
        str: 估计的训练时间
    """
    import math

    # 假设每批次处理时间（秒）
    if config.use_amp:
        batch_processing_time = 0.05  # 混合精度更快
    else:
        batch_processing_time = 0.1

    batches_per_epoch = math.ceil(dataset_size / config.batch_size)
    time_per_epoch = batches_per_epoch * batch_processing_time
    total_time_seconds = time_per_epoch * config.epochs

    # 转换为小时和分钟
    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)

    return f"{hours}小时{minutes}分钟"


if __name__ == "__main__":
    # 测试函数
    config = get_training_config("default")
    print(f"默认训练配置: {config}")

    kaggle_config = get_training_config("kaggle")
    print(f"Kaggle训练配置: {kaggle_config}")

    # 估计训练时间
    dataset_size = 10000  # 假设数据集有10000个样本
    training_time = estimate_training_time(config, dataset_size)
    print(f"估计训练时间: {training_time}")

    # 优化器配置
    optimizer_config = create_optimizer_config("adamw")
    print(f"AdamW优化器配置: {optimizer_config}")