"""
Point Transformer模型
基于注意力机制的点云分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

from .base_model import BaseModel


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, dim: int):
        """
        初始化位置编码

        Args:
            dim: 编码维度
        """
        super().__init__()
        self.dim = dim
        # 使用可学习的线性投影将3D坐标映射到dim维
        self.linear = nn.Linear(3, dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        计算位置编码

        Args:
            pos: 位置坐标，形状为 (B, N, 3)

        Returns:
            torch.Tensor: 位置编码，形状为 (B, N, dim)
        """
        # 使用线性投影
        return self.linear(pos)


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.1, attn_drop_rate: float = 0.1):
        """
        初始化Transformer块

        Args:
            dim: 特征维度
            num_heads: 注意力头数量
            mlp_ratio: MLP扩展比例
            qkv_bias: 是否在QKV投影中使用偏置
            drop_rate: Dropout率
            attn_drop_rate: 注意力Dropout率
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop_rate, drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop_rate)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征，形状为 (B, N, C)
            pos: 位置编码，形状为 (B, N, C)

        Returns:
            torch.Tensor: 输出特征
        """
        # 添加位置编码
        if pos is not None:
            x = x + pos

        # 自注意力
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop_rate: float = 0.1, proj_drop_rate: float = 0.1):
        """
        初始化多头注意力

        Args:
            dim: 特征维度
            num_heads: 注意力头数量
            qkv_bias: 是否在QKV投影中使用偏置
            attn_drop_rate: 注意力Dropout率
            proj_drop_rate: 投影Dropout率
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征，形状为 (B, N, C)

        Returns:
            torch.Tensor: 注意力输出
        """
        batch_size, num_points, dim = x.shape

        # 计算QKV
        qkv = self.qkv(x).reshape(batch_size, num_points, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_points, dim)

        # 投影
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, drop_rate: float = 0.1):
        """
        初始化MLP

        Args:
            in_dim: 输入维度
            hidden_dim: 隐藏层维度
            out_dim: 输出维度
            drop_rate: Dropout率
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征

        Returns:
            torch.Tensor: MLP输出
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PointTransformer(BaseModel):
    """Point Transformer点云分类模型"""

    def __init__(self, num_classes: int = 15, num_points: int = 1024, dim: int = 512,
                 depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.1, attn_drop_rate: float = 0.1):
        """
        初始化Point Transformer

        Args:
            num_classes: 类别数量
            num_points: 点云点数
            dim: 特征维度
            depth: Transformer块深度
            num_heads: 注意力头数量
            mlp_ratio: MLP扩展比例
            qkv_bias: 是否在QKV投影中使用偏置
            drop_rate: Dropout率
            attn_drop_rate: 注意力Dropout率
        """
        super().__init__(num_classes=num_classes)

        self.num_points = num_points
        self.dim = dim
        self.depth = depth

        # 输入嵌入层
        self.input_proj = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.BatchNorm1d(dim // 2) if num_points > 1 else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.BatchNorm1d(dim) if num_points > 1 else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(dim)

        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # 输出层
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(dim, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, N, 3)

        Returns:
            torch.Tensor: 分类得分，形状为 (B, num_classes)
        """
        batch_size, num_points, _ = x.shape

        # 输入投影
        x_reshaped = x.view(-1, 3)  # (B*N, 3)
        features = self.input_proj(x_reshaped)  # (B*N, dim)
        features = features.view(batch_size, num_points, -1)  # (B, N, dim)

        # 位置编码
        pos = self.pos_encoder(x)  # (B, N, dim)

        # Transformer块
        for block in self.blocks:
            features = block(features, pos)

        # 全局平均池化
        features = self.norm(features)
        global_features = torch.mean(features, dim=1)  # (B, dim)

        # 分类
        output = self.output_proj(global_features)  # (B, num_classes)

        return output

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

        # 输入投影
        x_reshaped = x.view(-1, 3)
        features = self.input_proj(x_reshaped)
        features = features.view(batch_size, num_points, -1)

        # 位置编码
        pos = self.pos_encoder(x)

        # 存储所有层的特征
        all_features = [features] if return_all else None

        # Transformer块
        for block in self.blocks:
            features = block(features, pos)
            if return_all:
                all_features.append(features)

        # 全局平均池化
        features = self.norm(features)
        global_features = torch.mean(features, dim=1)

        if return_all:
            return global_features, all_features
        else:
            return global_features


class SimplifiedPointTransformer(PointTransformer):
    """简化的Point Transformer（计算量更小）"""

    def __init__(self, num_classes: int = 15, num_points: int = 1024, dim: int = 256,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0):
        """
        初始化简化的Point Transformer

        Args:
            num_classes: 类别数量
            num_points: 点云点数
            dim: 特征维度
            depth: Transformer块深度
            num_heads: 注意力头数量
            mlp_ratio: MLP扩展比例
        """
        super().__init__(
            num_classes=num_classes,
            num_points=num_points,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )


def create_point_transformer(config: dict) -> PointTransformer:
    """
    根据配置创建Point Transformer

    Args:
        config: 配置字典

    Returns:
        PointTransformer: Point Transformer模型
    """
    # 默认配置
    default_config = {
        "num_classes": 15,
        "num_points": 1024,
        "dim": 512,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
    }

    # 更新配置
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    return PointTransformer(**config)


def test_point_transformer():
    """测试Point Transformer模型"""
    print("测试Point Transformer模型...")

    # 创建模型
    model = PointTransformer(
        num_classes=15,
        num_points=1024,
        dim=512,
        depth=6,
        num_heads=8
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
    features = model.extract_features(test_input)
    print(f"特征形状: {features.shape}")

    # 测试简化版本
    print("\n测试简化版本:")
    simple_model = SimplifiedPointTransformer(
        num_classes=15,
        num_points=1024,
        dim=256,
        depth=4,
        num_heads=4
    )

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
        model_path = os.path.join(temp_dir, "point_transformer.pth")

        # 保存模型
        model.save(model_path, epoch=0, metrics={"test_acc": 0.0})
        print(f"\n模型保存到: {model_path}")

        # 加载模型
        new_model = PointTransformer(num_classes=15, num_points=1024)
        loaded_info = new_model.load(model_path)
        print(f"加载信息: {loaded_info}")

        # 验证加载的模型
        with torch.no_grad():
            new_output = new_model(test_input)
            print(f"原始输出与加载后输出是否一致: {torch.allclose(output, new_output, rtol=1e-4)}")

    print("\nPoint Transformer模型测试通过!")


if __name__ == "__main__":
    test_point_transformer()