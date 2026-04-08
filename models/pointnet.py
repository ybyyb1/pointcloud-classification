"""
PointNet模型
经典的点云分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from .base_model import BaseModel


class TNet(nn.Module):
    """变换网络（T-Net）"""

    def __init__(self, k: int = 3):
        """
        初始化T-Net

        Args:
            k: 输入维度（对于点云坐标，k=3）
        """
        super().__init__()
        self.k = k

        # 特征提取网络
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # 全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 初始化变换矩阵为单位矩阵
        self.fc3.weight.data.zero_()
        self.fc3.bias.data = torch.eye(k).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, k, N)

        Returns:
            torch.Tensor: 变换矩阵，形状为 (B, k, k)
        """
        batch_size = x.size(0)

        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # 全连接层
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 重塑为变换矩阵
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        transformation = x.view(-1, self.k, self.k) + identity

        return transformation


class PointNet(BaseModel):
    """PointNet点云分类模型"""

    def __init__(self, num_classes: int = 15, input_channels: int = 3,
                 use_tnet: bool = True, dropout: float = 0.3):
        """
        初始化PointNet

        Args:
            num_classes: 类别数量
            input_channels: 输入通道数
            use_tnet: 是否使用T-Net
            dropout: Dropout率
        """
        super().__init__(num_classes=num_classes, input_channels=input_channels)

        self.use_tnet = use_tnet
        self.dropout = dropout

        # 输入变换网络
        if use_tnet:
            self.input_tnet = TNet(k=input_channels)

        # 特征提取网络
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 特征变换网络
        if use_tnet:
            self.feature_tnet = TNet(k=64)

        # 分类网络
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(p=dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, N, C)

        Returns:
            torch.Tensor: 分类得分，形状为 (B, num_classes)
        """
        # 转换为 (B, C, N) 格式
        x = x.transpose(2, 1)  # (B, C, N)

        batch_size, num_channels, num_points = x.shape

        # 输入变换
        if self.use_tnet:
            input_transform = self.input_tnet(x)  # (B, C, C)
            x = torch.bmm(input_transform, x)  # (B, C, N)

        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        # 特征变换
        if self.use_tnet:
            feature_transform = self.feature_tnet(x)  # (B, 64, 64)
            x = torch.bmm(feature_transform, x)  # (B, 64, N)

        # 后续卷积层
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))  # (B, 1024, N)

        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.view(-1, 1024)  # (B, 1024)

        # 分类网络
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout_layer(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout_layer(x)
        x = self.fc3(x)

        return x

    def extract_features(self, x: torch.Tensor, return_transform: bool = False) -> torch.Tensor:
        """
        提取特征

        Args:
            x: 输入点云
            return_transform: 是否返回变换矩阵

        Returns:
            torch.Tensor: 提取的特征
        """
        # 转换为 (B, C, N) 格式
        x_input = x.transpose(2, 1)

        input_transform = None
        feature_transform = None

        # 输入变换
        if self.use_tnet:
            input_transform = self.input_tnet(x_input)
            x = torch.bmm(input_transform, x_input)
        else:
            x = x_input

        # 第一层卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 特征变换
        if self.use_tnet:
            feature_transform = self.feature_tnet(x)
            x = torch.bmm(feature_transform, x)

        # 后续卷积层
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # 全局最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if return_transform:
            return x, input_transform, feature_transform
        else:
            return x


class PointNetClassifier(PointNet):
    """简化的PointNet分类器"""

    def __init__(self, num_classes: int = 15, dropout: float = 0.3):
        """
        初始化简化的PointNet

        Args:
            num_classes: 类别数量
            dropout: Dropout率
        """
        super().__init__(
            num_classes=num_classes,
            input_channels=3,
            use_tnet=False,  # 简化版本不使用T-Net
            dropout=dropout
        )


def create_pointnet(config: dict) -> PointNet:
    """
    根据配置创建PointNet

    Args:
        config: 配置字典

    Returns:
        PointNet: PointNet模型
    """
    # 默认配置
    default_config = {
        "num_classes": 15,
        "input_channels": 3,
        "use_tnet": True,
        "dropout": 0.3,
    }

    # 更新配置
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # 过滤配置参数，只保留PointNet构造函数接受的参数
    import inspect
    sig = inspect.signature(PointNet.__init__)
    valid_params = list(sig.parameters.keys())

    filtered_config = {}
    for key, value in config.items():
        if key in valid_params and key != 'self':
            filtered_config[key] = value
        elif key != 'self':
            # 警告但忽略未知参数
            print(f"警告: PointNet 忽略未知参数 '{key}'")

    return PointNet(**filtered_config)


def test_pointnet():
    """测试PointNet模型"""
    print("测试PointNet模型...")

    # 创建模型
    model = PointNet(
        num_classes=15,
        use_tnet=True,
        dropout=0.3
    )

    # 测试参数
    params = model.get_parameters()
    print(f"模型参数: {params}")

    total_params, trainable_params = model.count_parameters()
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试前向传播
    batch_size = 4
    num_points = 1024
    test_input = torch.randn(batch_size, num_points, 3)

    print(f"\n输入形状: {test_input.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")

    # 测试特征提取
    features, input_transform, feature_transform = model.extract_features(test_input, return_transform=True)
    print(f"特征形状: {features.shape}")
    if input_transform is not None:
        print(f"输入变换矩阵形状: {input_transform.shape}")
    if feature_transform is not None:
        print(f"特征变换矩阵形状: {feature_transform.shape}")

    # 测试简化版本
    print("\n测试简化版本:")
    simple_model = PointNetClassifier(num_classes=15, dropout=0.3)

    total_params_simple, trainable_params_simple = simple_model.count_parameters()
    print(f"简化版总参数: {total_params_simple:,}")
    print(f"简化版可训练参数: {trainable_params_simple:,}")

    with torch.no_grad():
        simple_output = simple_model(test_input)
        print(f"简化版输出形状: {simple_output.shape}")

    # 测试模型保存和加载
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "pointnet.pth")

        # 保存模型
        model.save(model_path, epoch=0, metrics={"test_acc": 0.0})
        print(f"\n模型保存到: {model_path}")

        # 加载模型
        new_model = PointNet(num_classes=15)
        loaded_info = new_model.load(model_path)
        print(f"加载信息: {loaded_info}")

        # 验证加载的模型
        with torch.no_grad():
            new_output = new_model(test_input)
            print(f"原始输出与加载后输出是否一致: {torch.allclose(output, new_output, rtol=1e-4)}")

    print("\nPointNet模型测试通过!")


if __name__ == "__main__":
    test_pointnet()