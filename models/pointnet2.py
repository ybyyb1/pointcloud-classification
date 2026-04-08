"""
PointNet++模型
层次化点云分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .base_model import BaseModel


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    计算两组点之间的欧氏距离平方

    Args:
        src: 源点云，形状为 (B, N, C)
        dst: 目标点云，形状为 (B, M, C)

    Returns:
        torch.Tensor: 距离矩阵，形状为 (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    根据索引从点云中选择点

    Args:
        points: 输入点云，形状为 (B, N, C)
        idx: 索引，形状为 (B, M) 或 (B, M, K)

    Returns:
        torch.Tensor: 选择的点，形状为 (B, M, C) 或 (B, M, K, C)
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    最远点采样

    Args:
        xyz: 点云坐标，形状为 (B, N, C)
        npoint: 采样点数

    Returns:
        torch.Tensor: 采样点索引，形状为 (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    球查询：在给定半径内查找邻居点

    Args:
        radius: 球半径
        nsample: 最大邻居数
        xyz: 所有点坐标，形状为 (B, N, C)
        new_xyz: 查询点坐标，形状为 (B, S, C)

    Returns:
        torch.Tensor: 邻居点索引，形状为 (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    采样和分组

    Args:
        npoint: 采样点数
        radius: 球半径
        nsample: 最大邻居数
        xyz: 点云坐标，形状为 (B, N, C)
        points: 点云特征，形状为 (B, N, D)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (新点坐标, 分组特征)
    """
    B, N, C = xyz.shape
    S = npoint

    # 最远点采样
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, C)

    # 球查询
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)

    # 相对坐标
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, C+D)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """PointNet集合抽象层"""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: List[int], group_all: bool = False):
        """
        初始化集合抽象层

        Args:
            npoint: 采样点数
            radius: 球半径
            nsample: 最大邻居数
            in_channel: 输入通道数
            mlp: MLP各层通道数
            group_all: 是否将所有点作为一个组
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            xyz: 点云坐标，形状为 (B, N, 3)
            points: 点云特征，形状为 (B, N, D)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (新点坐标, 新特征)
        """
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, C)
            if points is not None:
                new_points = torch.cat([xyz, points], dim=-1)
            else:
                new_points = xyz
            new_points = new_points.unsqueeze(2)  # (B, N, 1, C+D)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )  # (B, npoint, C), (B, npoint, nsample, C+D)

        # 转换为 Conv2d 需要的格式: (B, C, npoint, nsample)
        new_points = new_points.permute(0, 3, 1, 2)

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 最大池化
        new_points = torch.max(new_points, 3)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points


class PointNet2(BaseModel):
    """PointNet++点云分类模型"""

    def __init__(self, num_classes: int = 15,
                 sa_layers: List[Dict[str, Any]] = None):
        """
        初始化PointNet++

        Args:
            num_classes: 类别数量
            sa_layers: 集合抽象层配置
        """
        super().__init__(num_classes=num_classes)

        # 默认配置
        if sa_layers is None:
            sa_layers = [
                {"npoint": 512, "radius": 0.2, "nsample": 32, "mlp": [64, 64, 128]},
                {"npoint": 128, "radius": 0.4, "nsample": 64, "mlp": [128, 128, 256]},
                {"npoint": None, "radius": None, "nsample": None, "mlp": [256, 512, 1024], "group_all": True}
            ]

        self.sa_layers = sa_layers

        # 构建集合抽象层
        self.sa_modules = nn.ModuleList()
        in_channel = 3  # 初始只有XYZ坐标

        for i, sa_config in enumerate(sa_layers):
            npoint = sa_config.get("npoint")
            radius = sa_config.get("radius")
            nsample = sa_config.get("nsample")
            mlp = sa_config.get("mlp")
            group_all = sa_config.get("group_all", False)

            sa_module = PointNetSetAbstraction(
                npoint=npoint,
                radius=radius,
                nsample=nsample,
                in_channel=in_channel,
                mlp=mlp,
                group_all=group_all
            )
            self.sa_modules.append(sa_module)

            # 更新输入通道数
            if group_all:
                in_channel = mlp[-1]
            else:
                in_channel = mlp[-1] + 3  # 加上坐标信息

        # 分类网络
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.5)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, N, 3)

        Returns:
            torch.Tensor: 分类得分，形状为 (B, num_classes)
        """
        B, N, _ = x.shape

        # 集合抽象
        xyz = x
        points = None

        for i, sa_module in enumerate(self.sa_modules):
            xyz, points = sa_module(xyz, points)

        # 全局特征
        x = points.view(B, -1)  # (B, 1024)

        # 分类网络
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def extract_features(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        提取特征

        Args:
            x: 输入点云
            return_intermediate: 是否返回中间特征

        Returns:
            torch.Tensor: 提取的特征
        """
        B, N, _ = x.shape

        # 集合抽象
        xyz = x
        points = None
        intermediate_features = []

        for i, sa_module in enumerate(self.sa_modules):
            xyz, points = sa_module(xyz, points)
            if return_intermediate:
                intermediate_features.append((xyz, points))

        # 全局特征
        features = points.view(B, -1)

        if return_intermediate:
            return features, intermediate_features
        else:
            return features


class SimplifiedPointNet2(PointNet2):
    """简化的PointNet++"""

    def __init__(self, num_classes: int = 15):
        """
        初始化简化的PointNet++

        Args:
            num_classes: 类别数量
        """
        sa_layers = [
            {"npoint": 256, "radius": 0.2, "nsample": 16, "mlp": [32, 32, 64]},
            {"npoint": 64, "radius": 0.4, "nsample": 32, "mlp": [64, 64, 128]},
            {"npoint": None, "radius": None, "nsample": None, "mlp": [128, 256, 512], "group_all": True}
        ]

        super().__init__(num_classes=num_classes, sa_layers=sa_layers)


def create_pointnet2(config: dict) -> PointNet2:
    """
    根据配置创建PointNet++

    Args:
        config: 配置字典

    Returns:
        PointNet2: PointNet++模型
    """
    # 默认配置
    default_config = {
        "num_classes": 15,
        "sa_layers": None,
    }

    # 更新配置
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # 过滤配置参数，只保留PointNet2构造函数接受的参数
    import inspect
    sig = inspect.signature(PointNet2.__init__)
    valid_params = list(sig.parameters.keys())

    filtered_config = {}
    for key, value in config.items():
        if key in valid_params and key != 'self':
            filtered_config[key] = value
        elif key != 'self':
            # 警告但忽略未知参数
            print(f"警告: PointNet2 忽略未知参数 '{key}'")

    return PointNet2(**filtered_config)


def test_pointnet2():
    """测试PointNet++模型"""
    print("测试PointNet++模型...")

    # 创建模型
    model = PointNet2(num_classes=15)

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
    features, intermediate = model.extract_features(test_input, return_intermediate=True)
    print(f"特征形状: {features.shape}")
    print(f"中间特征数量: {len(intermediate)}")
    for i, (xyz, points) in enumerate(intermediate):
        print(f"  第{i}层: xyz形状={xyz.shape}, 特征形状={points.shape}")

    # 测试简化版本
    print("\n测试简化版本:")
    simple_model = SimplifiedPointNet2(num_classes=15)

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
        model_path = os.path.join(temp_dir, "pointnet2.pth")

        # 保存模型
        model.save(model_path, epoch=0, metrics={"test_acc": 0.0})
        print(f"\n模型保存到: {model_path}")

        # 加载模型
        new_model = PointNet2(num_classes=15)
        loaded_info = new_model.load(model_path)
        print(f"加载信息: {loaded_info}")

        # 验证加载的模型
        with torch.no_grad():
            new_output = new_model(test_input)
            print(f"原始输出与加载后输出是否一致: {torch.allclose(output, new_output, rtol=1e-4)}")

    print("\nPointNet++模型测试通过!")


if __name__ == "__main__":
    test_pointnet2()