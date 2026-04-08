"""
损失函数模块
包含各种损失函数的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Union
import warnings


class ClassificationLoss(nn.Module):
    """分类损失函数（支持多种类型）"""

    def __init__(self, loss_type: str = "cross_entropy", **kwargs):
        """
        初始化分类损失函数

        Args:
            loss_type: 损失函数类型
            **kwargs: 损失函数参数
        """
        super().__init__()
        self.loss_type = loss_type.lower()

        # 根据类型选择损失函数
        if self.loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(**kwargs)

        elif self.loss_type == "label_smoothing":
            smoothing = kwargs.get("smoothing", 0.1)
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)

        elif self.loss_type == "focal":
            alpha = kwargs.get("alpha", 0.25)
            gamma = kwargs.get("gamma", 2.0)
            self.loss_fn = FocalLoss(alpha=alpha, gamma=gamma)

        elif self.loss_type == "dice":
            smooth = kwargs.get("smooth", 1.0)
            self.loss_fn = DiceLoss(smooth=smooth)

        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        Args:
            inputs: 模型输出，形状为 (B, C)
            targets: 目标标签，形状为 (B,)

        Returns:
            torch.Tensor: 损失值
        """
        return self.loss_fn(inputs, targets)


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        """
        初始化标签平滑交叉熵

        Args:
            smoothing: 平滑因子
            reduction: 减少方式，"mean" 或 "sum"
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑交叉熵损失

        Args:
            x: 模型输出，形状为 (B, C)
            target: 目标标签，形状为 (B,)

        Returns:
            torch.Tensor: 损失值
        """
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss（用于类别不平衡）"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        初始化Focal Loss

        Args:
            alpha: 平衡因子
            gamma: 聚焦因子
            reduction: 减少方式，"mean" 或 "sum"
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            inputs: 模型输出，形状为 (B, C)
            targets: 目标标签，形状为 (B,)

        Returns:
            torch.Tensor: 损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss（用于分割，也适用于分类）"""

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        """
        初始化Dice Loss

        Args:
            smooth: 平滑因子
            reduction: 减少方式，"mean" 或 "sum"
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss

        Args:
            inputs: 模型输出，形状为 (B, C)
            targets: 目标标签，形状为 (B,)

        Returns:
            torch.Tensor: 损失值
        """
        # 将目标转换为one-hot编码
        num_classes = inputs.size(1)
        targets_onehot = F.one_hot(targets, num_classes).float()

        # 应用softmax获取概率
        inputs_softmax = F.softmax(inputs, dim=1)

        # 计算Dice系数
        intersection = torch.sum(inputs_softmax * targets_onehot, dim=(1, 2))
        union = torch.sum(inputs_softmax, dim=(1, 2)) + torch.sum(targets_onehot, dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class WeightedLoss(nn.Module):
    """加权损失函数"""

    def __init__(self, loss_fn: nn.Module, weights: Optional[torch.Tensor] = None):
        """
        初始化加权损失函数

        Args:
            loss_fn: 基础损失函数
            weights: 类别权重
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失

        Args:
            inputs: 模型输出
            targets: 目标标签

        Returns:
            torch.Tensor: 损失值
        """
        loss = self.loss_fn(inputs, targets)

        if self.weights is not None:
            # 应用类别权重
            weight_tensor = self.weights.to(inputs.device)
            class_weights = weight_tensor[targets]
            loss = loss * class_weights

        return loss.mean()


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        """
        初始化组合损失函数

        Args:
            losses: 损失函数列表
            weights: 权重列表
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

        if len(self.weights) != len(self.losses):
            raise ValueError("权重数量必须与损失函数数量相同")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失

        Args:
            inputs: 模型输出
            targets: 目标标签

        Returns:
            torch.Tensor: 损失值
        """
        total_loss = 0.0
        loss_details = {}

        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
            loss_details[f"loss_{i}"] = loss.item()

        return total_loss


class ContrastiveLoss(nn.Module):
    """对比损失（用于度量学习）"""

    def __init__(self, margin: float = 1.0):
        """
        初始化对比损失

        Args:
            margin: 边界值
        """
        super().__init__()
        self.margin = margin

    def forward(self, features1: torch.Tensor, features2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失

        Args:
            features1: 特征1，形状为 (B, D)
            features2: 特征2，形状为 (B, D)
            labels: 标签，相同为1，不同为0

        Returns:
            torch.Tensor: 损失值
        """
        # 计算欧氏距离
        distance = F.pairwise_distance(features1, features2, p=2)

        # 计算损失
        loss = labels * distance.pow(2) + (1 - labels) * F.relu(self.margin - distance).pow(2)

        return loss.mean()


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    创建损失函数

    Args:
        loss_config: 损失函数配置

    Returns:
        nn.Module: 损失函数对象
    """
    # 默认配置
    default_config = {
        "type": "cross_entropy",
        "smoothing": 0.1,
        "alpha": 0.25,
        "gamma": 2.0,
        "smooth": 1.0,
        "reduction": "mean",
    }

    # 更新配置
    config = {**default_config, **loss_config}
    loss_type = config["type"].lower()

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=config["reduction"])

    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(
            smoothing=config["smoothing"],
            reduction=config["reduction"]
        )

    elif loss_type == "focal":
        return FocalLoss(
            alpha=config["alpha"],
            gamma=config["gamma"],
            reduction=config["reduction"]
        )

    elif loss_type == "dice":
        return DiceLoss(
            smooth=config["smooth"],
            reduction=config["reduction"]
        )

    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def calculate_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        dataset: 数据集
        num_classes: 类别数量

    Returns:
        torch.Tensor: 类别权重
    """
    # 统计每个类别的样本数量
    class_counts = torch.zeros(num_classes)

    for _, label in dataset:
        class_counts[label] += 1

    # 计算权重（样本数越少，权重越高）
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    # 归一化
    class_weights = class_weights / class_weights.sum()

    return class_weights


def test_loss_functions():
    """测试损失函数模块"""
    print("测试损失函数模块...")

    # 创建测试数据
    batch_size = 4
    num_classes = 5

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print(f"输入形状: {inputs.shape}")
    print(f"目标形状: {targets.shape}")

    # 测试各种损失函数
    loss_types = ["cross_entropy", "label_smoothing", "focal", "dice"]

    for loss_type in loss_types:
        try:
            loss_fn = create_loss_function({"type": loss_type})

            loss = loss_fn(inputs, targets)
            print(f"{loss_type}: {loss.item():.4f}")

        except Exception as e:
            print(f"{loss_type}: 创建失败 - {e}")

    # 测试ClassificationLoss包装器
    print("\n测试ClassificationLoss包装器:")
    for loss_type in ["cross_entropy", "label_smoothing"]:
        try:
            classification_loss = ClassificationLoss(loss_type=loss_type, smoothing=0.1)
            loss = classification_loss(inputs, targets)
            print(f"ClassificationLoss({loss_type}): {loss.item():.4f}")
        except Exception as e:
            print(f"ClassificationLoss({loss_type}): 失败 - {e}")

    # 测试加权损失函数
    print("\n测试加权损失函数:")
    try:
        # 创建模拟类别权重
        weights = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
        weighted_loss = WeightedLoss(nn.CrossEntropyLoss(), weights=weights)

        loss = weighted_loss(inputs, targets)
        print(f"WeightedLoss: {loss.item():.4f}")
    except Exception as e:
        print(f"加权损失函数失败: {e}")

    # 测试组合损失函数
    print("\n测试组合损失函数:")
    try:
        ce_loss = nn.CrossEntropyLoss()
        label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=0.1)

        combined_loss = CombinedLoss(
            losses=[ce_loss, label_smoothing_loss],
            weights=[0.7, 0.3]
        )

        loss = combined_loss(inputs, targets)
        print(f"CombinedLoss: {loss.item():.4f}")
    except Exception as e:
        print(f"组合损失函数失败: {e}")

    # 测试对比损失
    print("\n测试对比损失:")
    try:
        features1 = torch.randn(batch_size, 128)
        features2 = torch.randn(batch_size, 128)
        labels = torch.randint(0, 2, (batch_size,)).float()  # 0或1

        contrastive_loss = ContrastiveLoss(margin=1.0)
        loss = contrastive_loss(features1, features2, labels)
        print(f"ContrastiveLoss: {loss.item():.4f}")
    except Exception as e:
        print(f"对比损失失败: {e}")

    # 测试类别权重计算
    print("\n测试类别权重计算:")
    try:
        # 创建模拟数据集
        class MockDataset:
            def __init__(self):
                self.data = [(torch.randn(10), i % 5) for i in range(100)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = MockDataset()
        class_weights = calculate_class_weights(dataset, num_classes=5)
        print(f"类别权重: {class_weights.tolist()}")
    except Exception as e:
        print(f"类别权重计算失败: {e}")

    print("\n损失函数模块测试通过!")


if __name__ == "__main__":
    test_loss_functions()