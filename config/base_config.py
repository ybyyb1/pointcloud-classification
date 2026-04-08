"""
基础配置文件模块
定义系统的所有配置类
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class DatasetType(Enum):
    """数据集类型枚举"""
    SCANOBJECTNN = "scanobjectnn"
    S3DIS = "s3dis"
    STANFORD3D = "stanford3d"
    CUSTOM = "custom"


class ModelType(Enum):
    """模型类型枚举"""
    POINT_TRANSFORMER = "point_transformer"
    POINTNET = "pointnet"
    POINTNET2 = "pointnet2"
    DGCNN = "dgcnn"


@dataclass
class DatasetConfig:
    """数据集配置"""
    # 基础配置
    dataset_type: DatasetType = DatasetType.SCANOBJECTNN
    data_dir: str = "./data"
    num_points: int = 1024  # 点云采样点数
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True

    # 数据增强配置
    use_augmentation: bool = True
    rotation_range: Tuple[float, float] = (0, 360)  # 旋转角度范围
    translation_range: Tuple[float, float] = (-0.2, 0.2)  # 平移范围
    scale_range: Tuple[float, float] = (0.8, 1.2)  # 缩放范围
    jitter_std: float = 0.01  # 抖动标准差

    # ScanObjectNN特定配置
    scanobjectnn_version: str = "main_split"  # "main_split" 或 "pb_t50_rs_split"
    scanobjectnn_url: str = "https://github.com/hkust-vgd/scanobjectnn/raw/master/h5_files/main_split"

    # S3DIS特定配置
    s3dis_area: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    s3dis_url: str = "http://buildingparser.stanford.edu/dataset"
    s3dis_classes_to_include: List[str] = field(default_factory=lambda: [
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter"
    ])

    # Stanford3D特定配置
    stanford3d_areas: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    stanford3d_classes_to_include: List[str] = field(default_factory=lambda: [
        "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
        "door", "floor", "sofa", "stairs", "table", "wall", "window"
    ])

    # 数据集分割
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType = ModelType.POINT_TRANSFORMER
    num_classes: int = 15  # ScanObjectNN有15个类别
    input_channels: int = 3  # XYZ坐标

    # Point-transformer配置
    point_transformer_dim: int = 512
    point_transformer_depth: int = 6
    point_transformer_heads: int = 8
    point_transformer_mlp_ratio: float = 4.0
    point_transformer_qkv_bias: bool = True
    point_transformer_drop_rate: float = 0.1
    point_transformer_attn_drop_rate: float = 0.1

    # PointNet配置
    pointnet_mlp_layers: List[int] = field(default_factory=lambda: [64, 128, 1024])
    pointnet_use_tnet: bool = True
    pointnet_dropout: float = 0.3

    # PointNet++配置
    pointnet2_sa_layers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"npoint": 512, "radius": 0.2, "nsample": 32, "mlp": [64, 64, 128]},
        {"npoint": 128, "radius": 0.4, "nsample": 64, "mlp": [128, 128, 256]},
        {"npoint": None, "radius": None, "nsample": None, "mlp": [256, 512, 1024]}
    ])

    # DGCNN配置
    dgcnn_k: int = 20
    dgcnn_emb_dims: int = 1024
    dgcnn_dropout: float = 0.5

    # 通用模型配置
    use_batch_norm: bool = True
    activation: str = "relu"  # "relu", "leaky_relu", "gelu"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练配置
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    momentum: float = 0.9

    # 优化器配置
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"

    # 学习率调度器配置
    cosine_t_max: int = 300
    step_milestones: List[int] = field(default_factory=lambda: [100, 200])
    step_gamma: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 10

    # 损失函数配置
    loss_function: str = "cross_entropy"  # "cross_entropy", "label_smoothing"
    label_smoothing: float = 0.1

    # 训练策略
    use_amp: bool = True  # 自动混合精度
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0
    early_stopping_patience: int = 20

    # 检查点配置
    save_checkpoint_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    best_model_metric: str = "val_accuracy"

    # Kaggle特定配置
    kaggle_output_dir: str = "/kaggle/working"
    kaggle_gpu_memory_limit: Optional[float] = None  # GB


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "./checkpoints/best_model.pth"
    batch_size: int = 16
    use_gpu: bool = True
    gpu_id: int = 0

    # 后处理配置
    confidence_threshold: float = 0.5
    top_k: int = 3

    # 可视化配置
    visualize_results: bool = True
    visualization_dir: str = "./visualizations"
    save_pointclouds: bool = False


@dataclass
class UIConfig:
    """用户界面配置"""
    # CLI配置
    cli_enabled: bool = True
    cli_theme: str = "dark"  # "dark", "light"
    cli_show_progress: bool = True
    cli_progress_style: str = "bar"  # "bar", "spinner"

    # 桌面应用配置（可选）
    desktop_enabled: bool = False
    desktop_theme: str = "default"
    desktop_window_size: Tuple[int, int] = (1200, 800)

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/system.log"
    console_log: bool = True


@dataclass
class SystemConfig:
    """系统总配置"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # 系统配置
    project_name: str = "PointCloud Classification System"
    version: str = "1.0.0"
    author: str = ""
    description: str = "基于点云数据的室内场景三维物体分类系统"

    # 路径配置
    root_dir: str = "."
    data_dir: str = "./data"
    model_dir: str = "./models"
    output_dir: str = "./output"
    log_dir: str = "./logs"

    # 实验配置
    experiment_name: str = "exp_001"
    experiment_tags: List[str] = field(default_factory=list)
    use_wandb: bool = False
    wandb_project: str = "pointcloud-classification"

    def __post_init__(self):
        """后初始化处理，确保路径存在"""
        import os
        # 确保所有目录存在
        directories = [
            self.data_dir,
            self.model_dir,
            self.output_dir,
            self.log_dir,
            self.training.checkpoint_dir,
            self.inference.visualization_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，如果为None则使用默认配置

    Returns:
        SystemConfig: 系统配置对象
    """
    import yaml
    import json

    if config_path is None:
        return SystemConfig()

    # 根据文件扩展名选择加载方式
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")

    # 递归地将字典转换为配置对象
    def dict_to_config(d: Dict[str, Any]) -> SystemConfig:
        # 这里简化处理，实际应该递归转换嵌套的dataclass
        return SystemConfig(**d)

    return dict_to_config(config_dict)


def save_config(config: SystemConfig, config_path: str):
    """
    保存配置到文件

    Args:
        config: 系统配置对象
        config_path: 配置文件路径
    """
    import yaml
    import json
    from dataclasses import asdict

    # 将dataclass转换为字典
    config_dict = asdict(config)

    # 根据文件扩展名选择保存方式
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    elif config_path.endswith('.json'):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")


if __name__ == "__main__":
    # 测试配置
    config = SystemConfig()
    print("默认配置创建成功")
    print(f"项目名称: {config.project_name}")
    print(f"版本: {config.version}")

    # 保存示例配置
    save_config(config, "config/example_config.yaml")
    print("示例配置已保存到 config/example_config.yaml")