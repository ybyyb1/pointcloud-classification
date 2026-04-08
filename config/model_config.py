"""
模型配置文件
提供模型相关的配置和工具函数
"""

from typing import List, Dict, Any, Optional
from .base_config import ModelConfig, ModelType


# 预定义的模型配置
POINT_TRANSFORMER_CONFIG = ModelConfig(
    model_type=ModelType.POINT_TRANSFORMER,
    num_classes=15,
    point_transformer_dim=512,
    point_transformer_depth=6,
    point_transformer_heads=8,
    point_transformer_mlp_ratio=4.0,
    point_transformer_drop_rate=0.1,
    use_batch_norm=True,
    activation="gelu"
)

POINTNET_CONFIG = ModelConfig(
    model_type=ModelType.POINTNET,
    num_classes=15,
    pointnet_mlp_layers=[64, 128, 1024],
    pointnet_use_tnet=True,
    pointnet_dropout=0.3,
    use_batch_norm=True,
    activation="relu"
)

POINTNET2_CONFIG = ModelConfig(
    model_type=ModelType.POINTNET2,
    num_classes=15,
    pointnet2_sa_layers=[
        {"npoint": 512, "radius": 0.2, "nsample": 32, "mlp": [64, 64, 128]},
        {"npoint": 128, "radius": 0.4, "nsample": 64, "mlp": [128, 128, 256]},
        {"npoint": None, "radius": None, "nsample": None, "mlp": [256, 512, 1024]}
    ],
    use_batch_norm=True,
    activation="relu"
)

DGCNN_CONFIG = ModelConfig(
    model_type=ModelType.DGCNN,
    num_classes=15,
    dgcnn_k=20,
    dgcnn_emb_dims=1024,
    dgcnn_dropout=0.5,
    use_batch_norm=True,
    activation="leaky_relu"
)


def get_model_config(model_name: str) -> ModelConfig:
    """
    根据模型名称获取配置

    Args:
        model_name: 模型名称，支持 "point_transformer", "pointnet", "pointnet2", "dgcnn"

    Returns:
        ModelConfig: 模型配置对象
    """
    model_name = model_name.lower()
    if model_name in ["point_transformer", "point-transformer", "transformer"]:
        return POINT_TRANSFORMER_CONFIG
    elif model_name == "pointnet":
        return POINTNET_CONFIG
    elif model_name in ["pointnet2", "pointnet++"]:
        return POINTNET2_CONFIG
    elif model_name == "dgcnn":
        return DGCNN_CONFIG
    else:
        raise ValueError(f"未知的模型: {model_name}")


def get_model_parameters(model_name: str) -> Dict[str, Any]:
    """
    获取模型的参数量估计

    Args:
        model_name: 模型名称

    Returns:
        Dict[str, Any]: 参数字典，包含总参数和可训练参数
    """
    import torch
    from models.model_factory import create_model

    config = get_model_config(model_name)
    model = create_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_config": config
    }


def print_model_summary(model_name: str):
    """
    打印模型摘要

    Args:
        model_name: 模型名称
    """
    params = get_model_parameters(model_name)

    print(f"模型名称: {params['model_name']}")
    print(f"总参数: {params['total_parameters']:,}")
    print(f"可训练参数: {params['trainable_parameters']:,}")
    print(f"模型配置: {params['model_config']}")


if __name__ == "__main__":
    # 测试函数
    config = get_model_config("point_transformer")
    print(f"PointTransformer配置: {config}")

    config2 = get_model_config("pointnet")
    print(f"PointNet配置: {config2}")

    # 打印模型摘要
    print("\n模型摘要:")
    print_model_summary("point_transformer")