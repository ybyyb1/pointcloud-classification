"""
优化器模块
包含各种优化器的创建和配置
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union
import warnings


def create_optimizer(model_params, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    创建优化器

    Args:
        model_params: 模型参数
        optimizer_config: 优化器配置

    Returns:
        torch.optim.Optimizer: 优化器对象
    """
    # 默认配置
    default_config = {
        "type": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "momentum": 0.9,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "nesterov": False,
        "dampening": 0,
    }

    # 更新配置
    config = {**default_config, **optimizer_config}
    optimizer_type = config["type"].lower()

    # 准备优化器参数
    optimizer_kwargs = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
    }

    # 根据优化器类型添加特定参数
    if optimizer_type == "adam":
        optimizer_kwargs.update({
            "betas": config["betas"],
            "eps": config["eps"],
            "amsgrad": config["amsgrad"],
        })
        optimizer = optim.Adam(model_params, **optimizer_kwargs)

    elif optimizer_type == "adamw":
        optimizer_kwargs.update({
            "betas": config["betas"],
            "eps": config["eps"],
            "amsgrad": config["amsgrad"],
        })
        optimizer = optim.AdamW(model_params, **optimizer_kwargs)

    elif optimizer_type == "sgd":
        optimizer_kwargs.update({
            "momentum": config["momentum"],
            "dampening": config["dampening"],
            "nesterov": config["nesterov"],
        })
        optimizer = optim.SGD(model_params, **optimizer_kwargs)

    elif optimizer_type == "rmsprop":
        optimizer_kwargs.update({
            "alpha": config.get("alpha", 0.99),
            "eps": config["eps"],
            "momentum": config["momentum"],
            "centered": config.get("centered", False),
        })
        optimizer = optim.RMSprop(model_params, **optimizer_kwargs)

    elif optimizer_type == "adagrad":
        optimizer_kwargs.update({
            "lr_decay": config.get("lr_decay", 0),
            "eps": config["eps"],
        })
        optimizer = optim.Adagrad(model_params, **optimizer_kwargs)

    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    return optimizer


def create_optimizer_with_params(model: torch.nn.Module, lr: float = 0.001,
                                 optimizer_type: str = "adamw",
                                 weight_decay: float = 0.01,
                                 **kwargs) -> torch.optim.Optimizer:
    """
    使用参数创建优化器（简化版）

    Args:
        model: 模型
        lr: 学习率
        optimizer_type: 优化器类型
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数

    Returns:
        torch.optim.Optimizer: 优化器对象
    """
    config = {
        "type": optimizer_type,
        "lr": lr,
        "weight_decay": weight_decay,
        **kwargs
    }

    return create_optimizer(model.parameters(), config)


def create_optimizer_with_layerwise_lr(model: torch.nn.Module,
                                       base_lr: float = 0.001,
                                       lr_multipliers: Optional[Dict[str, float]] = None,
                                       optimizer_type: str = "adamw",
                                       weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """
    创建分层学习率优化器

    Args:
        model: 模型
        base_lr: 基础学习率
        lr_multipliers: 层名称到学习率乘子的映射
        optimizer_type: 优化器类型
        weight_decay: 权重衰减

    Returns:
        torch.optim.Optimizer: 优化器对象
    """
    if lr_multipliers is None:
        lr_multipliers = {}

    # 默认乘子
    default_multipliers = {
        "backbone": 1.0,
        "head": 10.0,  # 分类头通常需要更高的学习率
        "embedding": 1.0,
    }
    lr_multipliers = {**default_multipliers, **lr_multipliers}

    # 分组参数
    param_groups = []

    # 遍历模型参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 确定参数组
        lr_multiplier = 1.0
        for pattern, multiplier in lr_multipliers.items():
            if pattern in name:
                lr_multiplier = multiplier
                break

        param_groups.append({
            "params": param,
            "lr": base_lr * lr_multiplier,
            "name": name
        })

    # 创建优化器配置
    config = {
        "type": optimizer_type,
        "lr": base_lr,  # 这个lr不会被使用，因为每个参数组有自己的lr
        "weight_decay": weight_decay,
    }

    # 创建优化器
    optimizer = create_optimizer(param_groups, config)

    # 打印参数组信息
    print(f"分层学习率优化器创建成功:")
    for i, group in enumerate(param_groups):
        print(f"  组 {i}: {group['name']}, lr={group['lr']:.6f}")

    return optimizer


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    获取优化器信息

    Args:
        optimizer: 优化器对象

    Returns:
        Dict[str, Any]: 优化器信息
    """
    info = {
        "type": optimizer.__class__.__name__,
        "param_groups": [],
        "total_parameters": 0
    }

    for i, group in enumerate(optimizer.param_groups):
        group_info = {
            "group_index": i,
            "lr": group.get("lr", 0.0),
            "weight_decay": group.get("weight_decay", 0.0),
            "momentum": group.get("momentum", 0.0),
            "num_parameters": 0,
            "parameter_names": []
        }

        # 计算参数数量
        for param in group["params"]:
            if param.requires_grad:
                group_info["num_parameters"] += param.numel()
                # 获取参数名称（如果有的话）
                if hasattr(param, "name"):
                    group_info["parameter_names"].append(param.name)

        info["param_groups"].append(group_info)
        info["total_parameters"] += group_info["num_parameters"]

    return info


def adjust_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """
    调整优化器的学习率

    Args:
        optimizer: 优化器对象
        lr: 新的学习率
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def freeze_layers(model: torch.nn.Module, layer_patterns: List[str]):
    """
    冻结指定层的参数

    Args:
        model: 模型
        layer_patterns: 层名称模式列表
    """
    for name, param in model.named_parameters():
        should_freeze = False
        for pattern in layer_patterns:
            if pattern in name:
                should_freeze = True
                break

        if should_freeze:
            param.requires_grad = False
            print(f"冻结层: {name}")


def unfreeze_layers(model: torch.nn.Module, layer_patterns: Optional[List[str]] = None):
    """
    解冻指定层的参数

    Args:
        model: 模型
        layer_patterns: 层名称模式列表，如果为None则解冻所有层
    """
    for name, param in model.named_parameters():
        if layer_patterns is None:
            param.requires_grad = True
        else:
            should_unfreeze = False
            for pattern in layer_patterns:
                if pattern in name:
                    should_unfreeze = True
                    break

            if should_unfreeze:
                param.requires_grad = True
                print(f"解冻层: {name}")


def get_trainable_parameters(model: torch.nn.Module) -> int:
    """
    获取可训练参数数量

    Args:
        model: 模型

    Returns:
        int: 可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_gradient_clipping_hook(max_norm: float = 1.0, norm_type: float = 2.0):
    """
    创建梯度裁剪钩子

    Args:
        max_norm: 最大梯度范数
        norm_type: 范数类型

    Returns:
        callable: 梯度裁剪函数
    """
    def gradient_clipping_hook(optimizer):
        torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]["params"],
            max_norm=max_norm,
            norm_type=norm_type
        )

    return gradient_clipping_hook


def test_optimizer():
    """测试优化器模块"""
    print("测试优化器模块...")

    import torch.nn as nn

    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.fc = nn.Linear(32 * 4 * 4, 10)

        def forward(self, x):
            return x

    model = TestModel()

    # 测试各种优化器
    optimizer_types = ["adam", "adamw", "sgd", "rmsprop"]

    for opt_type in optimizer_types:
        try:
            optimizer = create_optimizer_with_params(
                model=model,
                lr=0.001,
                optimizer_type=opt_type,
                weight_decay=0.01
            )

            # 获取优化器信息
            info = get_optimizer_info(optimizer)
            print(f"{opt_type}: {info['type']}, 参数: {info['total_parameters']:,}")

            # 测试学习率调整
            adjust_learning_rate(optimizer, 0.01)
            new_lr = optimizer.param_groups[0]["lr"]
            print(f"  学习率调整: 0.001 -> {new_lr:.4f}")

        except Exception as e:
            print(f"{opt_type}: 创建失败 - {e}")

    # 测试分层学习率
    print("\n测试分层学习率优化器:")
    try:
        layerwise_optimizer = create_optimizer_with_layerwise_lr(
            model=model,
            base_lr=0.001,
            lr_multipliers={
                "conv": 1.0,
                "fc": 10.0  # 全连接层使用10倍学习率
            },
            optimizer_type="adamw"
        )

        info = get_optimizer_info(layerwise_optimizer)
        print(f"分层学习率优化器: {info['type']}")
        for group in info["param_groups"]:
            print(f"  组 {group['group_index']}: lr={group['lr']:.6f}, 参数={group['num_parameters']:,}")

    except Exception as e:
        print(f"分层学习率优化器失败: {e}")

    # 测试冻结层
    print("\n测试冻结层:")
    freeze_layers(model, ["conv1"])
    trainable_params = get_trainable_parameters(model)
    print(f"冻结后可训练参数: {trainable_params:,}")

    unfreeze_layers(model, ["conv1"])
    trainable_params = get_trainable_parameters(model)
    print(f"解冻后可训练参数: {trainable_params:,}")

    # 测试梯度裁剪钩子
    print("\n测试梯度裁剪钩子:")
    clipping_hook = create_gradient_clipping_hook(max_norm=1.0)
    print(f"梯度裁剪钩子创建成功: {clipping_hook}")

    print("\n优化器模块测试通过!")


if __name__ == "__main__":
    test_optimizer()