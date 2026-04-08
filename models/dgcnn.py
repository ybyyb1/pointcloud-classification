"""
DGCNN模型
动态图卷积神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

from .base_model import BaseModel


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    计算K最近邻

    Args:
        x: 点云特征，形状为 (B, C, N)
        k: 最近邻数量

    Returns:
        torch.Tensor: K最近邻索引，形状为 (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)

    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    获取图特征

    Args:
        x: 点云特征，形状为 (B, C, N)
        k: 最近邻数量
        idx: 预计算的KNN索引（可选）

    Returns:
        torch.Tensor: 图特征，形状为 (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.shape
    device = x.device

    if idx is None:
        idx = knn(x, k)  # (B, N, k)

    # 获取邻居特征
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接中心点特征和邻居特征差异
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class EdgeConv(nn.Module):
    """边卷积层"""

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        """
        初始化边卷积层

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            k: 最近邻数量
        """
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征，形状为 (B, C, N)

        Returns:
            torch.Tensor: 输出特征，形状为 (B, C_out, N)
        """
        x = get_graph_feature(x, k=self.k)  # (B, 2*C, N, k)
        x = self.conv(x)  # (B, C_out, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C_out, N)

        return x


class DGCNN(BaseModel):
    """DGCNN点云分类模型"""

    def __init__(self, num_classes: int = 15, k: int = 20,
                 emb_dims: int = 1024, dropout: float = 0.5):
        """
        初始化DGCNN

        Args:
            num_classes: 类别数量
            k: 最近邻数量
            emb_dims: 嵌入维度
            dropout: Dropout率
        """
        super().__init__(num_classes=num_classes)

        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout

        # 边卷积层
        self.edge_conv1 = EdgeConv(3, 64, k=k)
        self.edge_conv2 = EdgeConv(64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, k=k)

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # 全局特征提取
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 分类网络
        self.conv6 = nn.Sequential(
            nn.Conv1d(emb_dims * 2, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv8 = nn.Conv1d(256, num_classes, kernel_size=1, bias=False)

        self.dropout_layer = nn.Dropout(p=dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, N, 3)

        Returns:
            torch.Tensor: 分类得分，形状为 (B, num_classes)
        """
        batch_size, num_points, _ = x.shape

        # 转换为 (B, C, N) 格式
        x = x.transpose(2, 1)  # (B, 3, N)

        # 边卷积层
        x1 = self.edge_conv1(x)  # (B, 64, N)
        x1 = self.bn1(x1)

        x2 = self.edge_conv2(x1)  # (B, 64, N)
        x2 = self.bn2(x2)

        x3 = self.edge_conv3(x2)  # (B, 128, N)
        x3 = self.bn3(x3)

        x4 = self.edge_conv4(x3)  # (B, 256, N)
        x4 = self.bn4(x4)

        # 拼接多层特征
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)

        # 全局特征
        x5 = self.conv5(x)  # (B, emb_dims, N)

        # 全局最大池化和平均池化
        x_max = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)  # (B, emb_dims)
        x_avg = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)  # (B, emb_dims)
        x_global = torch.cat((x_max, x_avg), 1)  # (B, emb_dims * 2)

        # 分类网络
        x = x_global.unsqueeze(-1)  # (B, emb_dims * 2, 1)
        x = self.conv6(x)  # (B, 512, 1)
        x = self.dropout_layer(x)
        x = self.conv7(x)  # (B, 256, 1)
        x = self.dropout_layer(x)
        x = self.conv8(x)  # (B, num_classes, 1)

        x = x.squeeze(-1)  # (B, num_classes)

        return x

    def extract_features(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        提取特征

        Args:
            x: 输入点云
            return_all: 是否返回所有层的特征

        Returns:
            torch.Tensor: 提取的特征
        """
        batch_size, num_points, _ = x.shape

        # 转换为 (B, C, N) 格式
        x_input = x.transpose(2, 1)

        # 边卷积层
        x1 = self.edge_conv1(x_input)
        x1 = self.bn1(x1)

        x2 = self.edge_conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.edge_conv3(x2)
        x3 = self.bn3(x3)

        x4 = self.edge_conv4(x3)
        x4 = self.bn4(x4)

        # 拼接多层特征
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        # 全局特征
        x5 = self.conv5(x_cat)

        # 全局最大池化和平均池化
        x_max = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x_global = torch.cat((x_max, x_avg), 1)

        if return_all:
            return x_global, [x1, x2, x3, x4, x_cat, x5]
        else:
            return x_global


class SimplifiedDGCNN(DGCNN):
    """简化的DGCNN"""

    def __init__(self, num_classes: int = 15):
        """
        初始化简化的DGCNN

        Args:
            num_classes: 类别数量
        """
        super().__init__(
            num_classes=num_classes,
            k=10,  # 减少最近邻数量
            emb_dims=512,  # 减少嵌入维度
            dropout=0.3  # 减少Dropout率
        )


def create_dgcnn(config: dict) -> DGCNN:
    """
    根据配置创建DGCNN

    Args:
        config: 配置字典

    Returns:
        DGCNN: DGCNN模型
    """
    # 默认配置
    default_config = {
        "num_classes": 15,
        "k": 20,
        "emb_dims": 1024,
        "dropout": 0.5,
    }

    # 更新配置
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # 过滤配置参数，只保留DGCNN构造函数接受的参数
    import inspect
    sig = inspect.signature(DGCNN.__init__)
    valid_params = list(sig.parameters.keys())

    filtered_config = {}
    for key, value in config.items():
        if key in valid_params and key != 'self':
            filtered_config[key] = value
        elif key != 'self':
            # 警告但忽略未知参数
            print(f"警告: DGCNN 忽略未知参数 '{key}'")

    return DGCNN(**filtered_config)


def test_dgcnn():
    """测试DGCNN模型"""
    print("测试DGCNN模型...")

    # 创建模型
    model = DGCNN(
        num_classes=15,
        k=20,
        emb_dims=1024,
        dropout=0.5
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
    features, all_features = model.extract_features(test_input, return_all=True)
    print(f"特征形状: {features.shape}")
    print(f"中间特征数量: {len(all_features)}")
    for i, feat in enumerate(all_features):
        print(f"  第{i}层特征形状: {feat.shape}")

    # 测试简化版本
    print("\n测试简化版本:")
    simple_model = SimplifiedDGCNN(num_classes=15)

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
        model_path = os.path.join(temp_dir, "dgcnn.pth")

        # 保存模型
        model.save(model_path, epoch=0, metrics={"test_acc": 0.0})
        print(f"\n模型保存到: {model_path}")

        # 加载模型
        new_model = DGCNN(num_classes=15)
        loaded_info = new_model.load(model_path)
        print(f"加载信息: {loaded_info}")

        # 验证加载的模型
        with torch.no_grad():
            new_output = new_model(test_input)
            print(f"原始输出与加载后输出是否一致: {torch.allclose(output, new_output, rtol=1e-4)}")

    print("\nDGCNN模型测试通过!")


if __name__ == "__main__":
    test_dgcnn()