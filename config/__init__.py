"""
配置模块
提供系统的所有配置类
"""

from .base_config import (
    SystemConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    UIConfig,
    DatasetType,
    ModelType,
    load_config,
    save_config
)

from .dataset_config import (
    SCANOBJECTNN_CONFIG,
    S3DIS_CONFIG,
    SCANOBJECTNN_CLASSES,
    SCANOBJECTNN_CLASS_TO_ID,
    SCANOBJECTNN_ID_TO_CLASS,
    S3DIS_TO_SCANOBJECTNN_MAPPING,
    get_dataset_config,
    get_class_names
)

from .model_config import (
    POINT_TRANSFORMER_CONFIG,
    POINTNET_CONFIG,
    POINTNET2_CONFIG,
    DGCNN_CONFIG,
    get_model_config,
    get_model_parameters,
    print_model_summary
)

from .training_config import (
    DEFAULT_TRAINING_CONFIG,
    KAGGLE_TRAINING_CONFIG,
    FAST_TRAINING_CONFIG,
    get_training_config,
    create_optimizer_config,
    create_scheduler_config,
    estimate_training_time
)

__all__ = [
    # 基础配置
    "SystemConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "UIConfig",
    "DatasetType",
    "ModelType",
    "load_config",
    "save_config",

    # 数据集配置
    "SCANOBJECTNN_CONFIG",
    "S3DIS_CONFIG",
    "SCANOBJECTNN_CLASSES",
    "SCANOBJECTNN_CLASS_TO_ID",
    "SCANOBJECTNN_ID_TO_CLASS",
    "S3DIS_TO_SCANOBJECTNN_MAPPING",
    "get_dataset_config",
    "get_class_names",

    # 模型配置
    "POINT_TRANSFORMER_CONFIG",
    "POINTNET_CONFIG",
    "POINTNET2_CONFIG",
    "DGCNN_CONFIG",
    "get_model_config",
    "get_model_parameters",
    "print_model_summary",

    # 训练配置
    "DEFAULT_TRAINING_CONFIG",
    "KAGGLE_TRAINING_CONFIG",
    "FAST_TRAINING_CONFIG",
    "get_training_config",
    "create_optimizer_config",
    "create_scheduler_config",
    "estimate_training_time",
]