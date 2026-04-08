"""
点云数据增强
包含各种点云数据增强方法
"""

import numpy as np
import random
from typing import Tuple, Optional, List
import torch


class PointCloudAugmentation:
    """点云数据增强类"""

    def __init__(self, config: Optional[dict] = None):
        """
        初始化点云增强

        Args:
            config: 增强配置字典
        """
        self.config = config or {}

        # 默认配置
        self.default_config = {
            "rotation_range": (0, 360),  # 旋转角度范围（度）
            "translation_range": (-0.2, 0.2),  # 平移范围
            "scale_range": (0.8, 1.2),  # 缩放范围
            "jitter_std": 0.01,  # 抖动标准差
            "jitter_clip": 0.05,  # 抖动裁剪值
            "dropout_ratio": 0.0,  # 丢弃点比例
            "random_rotation": True,
            "random_translation": True,
            "random_scaling": True,
            "random_jitter": True,
            "random_dropout": False,
        }

        # 更新配置
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        应用数据增强

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 增强后的点云
        """
        augmented_points = points.copy()

        # 随机旋转
        if self.config["random_rotation"]:
            augmented_points = self.random_rotation(augmented_points)

        # 随机平移
        if self.config["random_translation"]:
            augmented_points = self.random_translation(augmented_points)

        # 随机缩放
        if self.config["random_scaling"]:
            augmented_points = self.random_scaling(augmented_points)

        # 随机抖动
        if self.config["random_jitter"]:
            augmented_points = self.jitter_points(augmented_points)

        # 随机丢弃
        if self.config["random_dropout"] and self.config["dropout_ratio"] > 0:
            augmented_points = self.random_dropout(augmented_points)

        return augmented_points

    def random_rotation(self, points: np.ndarray) -> np.ndarray:
        """
        随机旋转点云

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 旋转后的点云
        """
        angle_range = self.config["rotation_range"]
        angle = np.random.uniform(angle_range[0], angle_range[1])

        # 转换为弧度
        angle_rad = np.deg2rad(angle)

        # 生成随机旋转轴
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)

        # 创建旋转矩阵
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        ux, uy, uz = axis

        rotation_matrix = np.array([
            [cos_theta + ux**2 * (1 - cos_theta),
             ux * uy * (1 - cos_theta) - uz * sin_theta,
             ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta,
             cos_theta + uy**2 * (1 - cos_theta),
             uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta,
             uz * uy * (1 - cos_theta) + ux * sin_theta,
             cos_theta + uz**2 * (1 - cos_theta)]
        ])

        # 应用旋转
        rotated_points = np.dot(points, rotation_matrix.T)

        return rotated_points

    def random_translation(self, points: np.ndarray) -> np.ndarray:
        """
        随机平移点云

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 平移后的点云
        """
        translation_range = self.config["translation_range"]
        translation = np.random.uniform(translation_range[0], translation_range[1], size=3)

        translated_points = points + translation

        return translated_points

    def random_scaling(self, points: np.ndarray) -> np.ndarray:
        """
        随机缩放点云

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 缩放后的点云
        """
        scale_range = self.config["scale_range"]
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scaled_points = points * scale

        return scaled_points

    def jitter_points(self, points: np.ndarray) -> np.ndarray:
        """
        添加随机抖动

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 抖动后的点云
        """
        std = self.config["jitter_std"]
        clip = self.config["jitter_clip"]

        noise = np.random.randn(*points.shape) * std
        noise = np.clip(noise, -clip, clip)

        jittered_points = points + noise

        return jittered_points

    def random_dropout(self, points: np.ndarray) -> np.ndarray:
        """
        随机丢弃点

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 丢弃点后的点云
        """
        dropout_ratio = self.config["dropout_ratio"]
        n_points = points.shape[0]
        n_keep = int(n_points * (1 - dropout_ratio))

        if n_keep < 3:  # 至少保留3个点
            n_keep = 3

        # 随机选择保留的点
        indices = np.random.choice(n_points, n_keep, replace=False)
        indices.sort()

        dropped_points = points[indices]

        return dropped_points

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        归一化点云（零均值，单位球内）

        Args:
            points: 输入点云，形状为 (N, 3)

        Returns:
            np.ndarray: 归一化的点云
        """
        # 零均值
        centroid = np.mean(points, axis=0)
        normalized = points - centroid

        # 缩放
        max_dist = np.max(np.sqrt(np.sum(normalized ** 2, axis=1)))
        if max_dist > 0:
            normalized = normalized / max_dist

        return normalized

    def rotate_around_axis(self, points: np.ndarray, axis: str, angle: float) -> np.ndarray:
        """
        绕指定轴旋转点云

        Args:
            points: 输入点云
            axis: 旋转轴，'x', 'y', 或 'z'
            angle: 旋转角度（度）

        Returns:
            np.ndarray: 旋转后的点云
        """
        angle_rad = np.deg2rad(angle)

        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"无效的轴: {axis}，必须是 'x', 'y', 或 'z'")

        rotated_points = np.dot(points, rotation_matrix.T)

        return rotated_points

    def flip_points(self, points: np.ndarray, axis: str) -> np.ndarray:
        """
        沿指定轴翻转点云

        Args:
            points: 输入点云
            axis: 翻转轴，'x', 'y', 或 'z'

        Returns:
            np.ndarray: 翻转后的点云
        """
        flipped_points = points.copy()

        if axis == 'x':
            flipped_points[:, 0] = -flipped_points[:, 0]
        elif axis == 'y':
            flipped_points[:, 1] = -flipped_points[:, 1]
        elif axis == 'z':
            flipped_points[:, 2] = -flipped_points[:, 2]
        else:
            raise ValueError(f"无效的轴: {axis}，必须是 'x', 'y', 或 'z'")

        return flipped_points


def create_augmentation_pipeline(configs: List[dict]) -> List[PointCloudAugmentation]:
    """
    创建增强管道

    Args:
        configs: 增强配置列表

    Returns:
        List[PointCloudAugmentation]: 增强管道
    """
    pipeline = []
    for config in configs:
        pipeline.append(PointCloudAugmentation(config))
    return pipeline


def apply_augmentation_pipeline(points: np.ndarray,
                                pipeline: List[PointCloudAugmentation]) -> np.ndarray:
    """
    应用增强管道

    Args:
        points: 输入点云
        pipeline: 增强管道

    Returns:
        np.ndarray: 增强后的点云
    """
    augmented_points = points.copy()
    for augmenter in pipeline:
        augmented_points = augmenter(augmented_points)
    return augmented_points


def test_augmentation():
    """测试数据增强"""
    print("测试点云数据增强...")

    # 创建测试点云
    points = np.random.randn(100, 3)

    # 创建增强器
    config = {
        "rotation_range": (0, 180),
        "translation_range": (-0.1, 0.1),
        "scale_range": (0.9, 1.1),
        "jitter_std": 0.02,
        "random_rotation": True,
        "random_translation": True,
        "random_scaling": True,
        "random_jitter": True
    }

    augmenter = PointCloudAugmentation(config)

    # 测试各种增强
    print("原始点云形状:", points.shape)
    print("原始点云范围:")
    print("  X: [{:.3f}, {:.3f}]".format(points[:, 0].min(), points[:, 0].max()))
    print("  Y: [{:.3f}, {:.3f}]".format(points[:, 1].min(), points[:, 1].max()))
    print("  Z: [{:.3f}, {:.3f}]".format(points[:, 2].min(), points[:, 2].max()))

    # 测试单个增强
    rotated = augmenter.random_rotation(points)
    translated = augmenter.random_translation(points)
    scaled = augmenter.random_scaling(points)
    jittered = augmenter.jitter_points(points)
    normalized = augmenter.normalize_points(points)

    print("\n增强测试完成:")
    print("  旋转后形状:", rotated.shape)
    print("  平移后形状:", translated.shape)
    print("  缩放后形状:", scaled.shape)
    print("  抖动后形状:", jittered.shape)
    print("  归一化后形状:", normalized.shape)

    # 测试完整的增强管道
    augmented = augmenter(points)
    print("\n完整增强后点云形状:", augmented.shape)

    # 测试增强管道
    pipeline_configs = [
        {"random_rotation": True, "rotation_range": (0, 90)},
        {"random_scaling": True, "scale_range": (0.8, 1.2)},
        {"random_jitter": True, "jitter_std": 0.01}
    ]

    pipeline = create_augmentation_pipeline(pipeline_configs)
    pipeline_augmented = apply_augmentation_pipeline(points, pipeline)
    print("管道增强后点云形状:", pipeline_augmented.shape)

    print("\n数据增强测试通过!")


if __name__ == "__main__":
    test_augmentation()