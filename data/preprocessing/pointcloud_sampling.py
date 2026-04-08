"""
点云采样
包含各种点云采样方法
"""

import numpy as np
from typing import Optional, Tuple, List
import random


class PointCloudSampler:
    """点云采样类"""

    def __init__(self, method: str = "farthest", random_seed: Optional[int] = None):
        """
        初始化采样器

        Args:
            method: 采样方法，可选 "random", "farthest", "uniform", "voxel"
            random_seed: 随机种子
        """
        self.method = method
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def sample(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        采样点云

        Args:
            points: 原始点云，形状为 (N, 3)
            n_samples: 采样点数

        Returns:
            np.ndarray: 采样后的点云，形状为 (n_samples, 3)
        """
        n_points = points.shape[0]

        if n_points <= n_samples:
            # 如果原始点数小于等于目标点数，直接返回（可能需要重复）
            return self._handle_insufficient_points(points, n_samples)

        if self.method == "random":
            return self.random_sampling(points, n_samples)
        elif self.method == "farthest":
            return self.farthest_point_sampling(points, n_samples)
        elif self.method == "uniform":
            return self.uniform_sampling(points, n_samples)
        elif self.method == "voxel":
            return self.voxel_sampling(points, n_samples)
        else:
            raise ValueError(f"未知的采样方法: {self.method}")

    def _handle_insufficient_points(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        处理点数不足的情况

        Args:
            points: 原始点云
            n_samples: 目标采样点数

        Returns:
            np.ndarray: 采样后的点云
        """
        n_points = points.shape[0]

        if n_points == n_samples:
            return points
        elif n_points > n_samples:
            # 不应该进入这个分支
            return self.random_sampling(points, n_samples)
        else:
            # 点数不足，重复采样
            n_repeats = n_samples // n_points + 1
            repeated_points = np.tile(points, (n_repeats, 1))
            indices = np.random.choice(repeated_points.shape[0], n_samples, replace=False)
            return repeated_points[indices]

    def random_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        随机采样

        Args:
            points: 原始点云
            n_samples: 采样点数

        Returns:
            np.ndarray: 采样后的点云
        """
        indices = np.random.choice(points.shape[0], n_samples, replace=False)
        indices.sort()
        return points[indices]

    def farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        最远点采样（FPS）
        保证采样点的空间分布均匀

        Args:
            points: 原始点云
            n_samples: 采样点数

        Returns:
            np.ndarray: 采样后的点云
        """
        n_points = points.shape[0]

        # 初始化
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(n_points) * np.inf

        # 随机选择第一个点
        first_idx = np.random.randint(n_points)
        sampled_indices[0] = first_idx

        # 迭代选择最远点
        for i in range(1, n_samples):
            # 更新距离
            last_point = points[sampled_indices[i - 1]]
            new_distances = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, new_distances)

            # 选择距离最大的点
            next_idx = np.argmax(distances)
            sampled_indices[i] = next_idx

        sampled_indices.sort()
        return points[sampled_indices]

    def uniform_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        均匀采样（按顺序等间隔采样）

        Args:
            points: 原始点云
            n_samples: 采样点数

        Returns:
            np.ndarray: 采样后的点云
        """
        # 随机打乱点云
        shuffled_indices = np.random.permutation(points.shape[0])
        shuffled_points = points[shuffled_indices]

        # 等间隔采样
        step = max(1, len(shuffled_points) // n_samples)
        indices = np.arange(0, len(shuffled_points), step)[:n_samples]

        return shuffled_points[indices]

    def voxel_sampling(self, points: np.ndarray, n_samples: int,
                       voxel_size: Optional[float] = None) -> np.ndarray:
        """
        体素采样（降采样）

        Args:
            points: 原始点云
            n_samples: 目标采样点数（用于计算体素大小）
            voxel_size: 体素大小，如果为None则自动计算

        Returns:
            np.ndarray: 采样后的点云
        """
        if voxel_size is None:
            # 根据目标点数估计体素大小
            bounding_box = np.max(points, axis=0) - np.min(points, axis=0)
            volume = np.prod(bounding_box)
            voxel_size = (volume / n_samples) ** (1/3)

        # 计算体素索引
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # 使用字典存储体素中的点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # 从每个体素中随机选择一个点
        sampled_indices = []
        for voxel_points in voxel_dict.values():
            sampled_indices.append(np.random.choice(voxel_points))

        sampled_points = points[sampled_indices]

        # 如果采样点数不足，补充随机采样
        if len(sampled_points) < n_samples:
            remaining_indices = list(set(range(len(points))) - set(sampled_indices))
            additional_indices = np.random.choice(remaining_indices,
                                                  n_samples - len(sampled_points),
                                                  replace=False)
            sampled_points = np.vstack([sampled_points, points[additional_indices]])

        # 如果采样点数过多，随机选择
        elif len(sampled_points) > n_samples:
            sampled_points = self.random_sampling(sampled_points, n_samples)

        return sampled_points

    def stratified_sampling(self, points: np.ndarray, n_samples: int,
                            n_strata: int = 8) -> np.ndarray:
        """
        分层采样

        Args:
            points: 原始点云
            n_samples: 采样点数
            n_strata: 层数

        Returns:
            np.ndarray: 采样后的点云
        """
        # 计算每个维度的范围
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)

        # 创建分层
        sampled_points = []
        samples_per_stratum = n_samples // n_strata

        for i in range(n_strata):
            # 计算当前层的边界
            x_min = min_vals[0] + (max_vals[0] - min_vals[0]) * i / n_strata
            x_max = min_vals[0] + (max_vals[0] - min_vals[0]) * (i + 1) / n_strata

            # 选择在当前层内的点
            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max)
            stratum_points = points[mask]

            if len(stratum_points) > 0:
                # 从当前层中采样
                if len(stratum_points) <= samples_per_stratum:
                    sampled_points.append(stratum_points)
                else:
                    stratum_samples = self.random_sampling(stratum_points, samples_per_stratum)
                    sampled_points.append(stratum_samples)

        # 合并所有层的采样点
        if sampled_points:
            sampled_points = np.vstack(sampled_points)

            # 如果采样点数不足，补充随机采样
            if len(sampled_points) < n_samples:
                remaining = n_samples - len(sampled_points)
                additional = self.random_sampling(points, remaining)
                sampled_points = np.vstack([sampled_points, additional])

            # 如果采样点数过多，随机选择
            elif len(sampled_points) > n_samples:
                sampled_points = self.random_sampling(sampled_points, n_samples)

            return sampled_points
        else:
            # 如果没有点在任何层中，使用随机采样
            return self.random_sampling(points, n_samples)


def sample_pointcloud(points: np.ndarray, n_samples: int,
                      method: str = "farthest") -> np.ndarray:
    """
    快速采样点云

    Args:
        points: 原始点云
        n_samples: 采样点数
        method: 采样方法

    Returns:
        np.ndarray: 采样后的点云
    """
    sampler = PointCloudSampler(method)
    return sampler.sample(points, n_samples)


def random_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    随机采样点云

    Args:
        points: 原始点云，形状为 (N, 3) 或 (3, N)
        n_samples: 采样点数

    Returns:
        np.ndarray: 采样后的点云
    """
    # 如果points是(3, N)格式，转换为(N, 3)
    transposed = False
    if points.shape[0] == 3 and points.shape[1] > 3:
        points = points.T
        transposed = True

    sampler = PointCloudSampler("random")
    sampled = sampler.sample(points, n_samples)

    # 返回与输入相同的格式
    if transposed:
        return sampled.T
    return sampled


def compare_sampling_methods(points: np.ndarray, n_samples: int) -> dict:
    """
    比较不同采样方法

    Args:
        points: 原始点云
        n_samples: 采样点数

    Returns:
        dict: 各种采样方法的结果和统计信息
    """
    methods = ["random", "farthest", "uniform", "voxel"]
    results = {}

    for method in methods:
        sampler = PointCloudSampler(method)
        sampled_points = sampler.sample(points, n_samples)

        # 计算统计信息
        centroid = np.mean(sampled_points, axis=0)
        bounding_box = np.max(sampled_points, axis=0) - np.min(sampled_points, axis=0)
        density = n_samples / np.prod(bounding_box) if np.all(bounding_box > 0) else 0

        results[method] = {
            "points": sampled_points,
            "centroid": centroid,
            "bounding_box": bounding_box,
            "density": density,
            "shape": sampled_points.shape
        }

    return results


def test_sampling():
    """测试点云采样"""
    print("测试点云采样...")

    # 创建测试点云
    points = np.random.randn(1000, 3)
    n_samples = 100

    print(f"原始点云: {points.shape}")
    print(f"目标采样点数: {n_samples}")

    # 测试各种采样方法
    methods = ["random", "farthest", "uniform", "voxel"]

    for method in methods:
        print(f"\n{method} 采样:")

        sampler = PointCloudSampler(method)
        sampled_points = sampler.sample(points, n_samples)

        print(f"  采样后形状: {sampled_points.shape}")
        print(f"  采样后均值: {np.mean(sampled_points, axis=0)}")
        print(f"  采样后范围: [{np.min(sampled_points, axis=0)}, {np.max(sampled_points, axis=0)}]")

        # 计算采样质量指标
        # 1. 覆盖率：采样点覆盖的空间范围
        original_range = np.max(points, axis=0) - np.min(points, axis=0)
        sampled_range = np.max(sampled_points, axis=0) - np.min(sampled_points, axis=0)
        coverage = np.mean(sampled_range / original_range)
        print(f"  空间覆盖率: {coverage:.3f}")

        # 2. 均匀性：采样点的分布均匀程度
        from scipy.spatial import KDTree
        if len(sampled_points) > 1:
            tree = KDTree(sampled_points)
            distances, _ = tree.query(sampled_points, k=2)
            avg_min_distance = np.mean(distances[:, 1])
            print(f"  平均最近邻距离: {avg_min_distance:.4f}")

    # 测试分层采样
    print("\n分层采样:")
    sampler = PointCloudSampler("random")
    stratified_points = sampler.stratified_sampling(points, n_samples, n_strata=5)
    print(f"  采样后形状: {stratified_points.shape}")

    # 比较所有方法
    print("\n比较所有采样方法:")
    comparison = compare_sampling_methods(points, n_samples)

    for method, result in comparison.items():
        print(f"  {method}: 形状={result['shape']}, 质心={result['centroid']}")

    print("\n点云采样测试通过!")


if __name__ == "__main__":
    test_sampling()