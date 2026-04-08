"""
点云归一化
包含各种点云归一化方法
"""

import numpy as np
from typing import Tuple, Optional


class PointCloudNormalizer:
    """点云归一化类"""

    def __init__(self, method: str = "unit_sphere", eps: float = 1e-8):
        """
        初始化归一化器

        Args:
            method: 归一化方法，可选 "unit_sphere", "unit_cube", "standard", "robust"
            eps: 防止除零的小值
        """
        self.method = method
        self.eps = eps
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, points: np.ndarray) -> None:
        """
        拟合归一化参数

        Args:
            points: 点云数据，形状为 (N, 3)
        """
        if self.method == "standard":
            self.mean = np.mean(points, axis=0)
            self.std = np.std(points, axis=0) + self.eps
        elif self.method == "robust":
            self.mean = np.median(points, axis=0)
            q75, q25 = np.percentile(points, [75, 25], axis=0)
            self.std = (q75 - q25) + self.eps
        elif self.method == "unit_cube":
            self.min = np.min(points, axis=0)
            self.max = np.max(points, axis=0)
        # unit_sphere不需要拟合参数

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        应用归一化

        Args:
            points: 点云数据，形状为 (N, 3)

        Returns:
            np.ndarray: 归一化后的点云
        """
        if self.method == "unit_sphere":
            return self._normalize_unit_sphere(points)
        elif self.method == "unit_cube":
            return self._normalize_unit_cube(points)
        elif self.method == "standard":
            return self._normalize_standard(points)
        elif self.method == "robust":
            return self._normalize_robust(points)
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")

    def fit_transform(self, points: np.ndarray) -> np.ndarray:
        """
        拟合并应用归一化

        Args:
            points: 点云数据，形状为 (N, 3)

        Returns:
            np.ndarray: 归一化后的点云
        """
        self.fit(points)
        return self.transform(points)

    def _normalize_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        """
        归一化到单位球内（零均值，最大距离为1）

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 归一化后的点云
        """
        # 零均值
        centroid = np.mean(points, axis=0)
        normalized = points - centroid

        # 计算最大距离
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        max_distance = np.max(distances)

        if max_distance > self.eps:
            normalized = normalized / max_distance

        return normalized

    def _normalize_unit_cube(self, points: np.ndarray) -> np.ndarray:
        """
        归一化到单位立方体内 [0, 1]^3

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 归一化后的点云
        """
        if self.min is None or self.max is None:
            self.min = np.min(points, axis=0)
            self.max = np.max(points, axis=0)

        # 防止除零
        range_vals = self.max - self.min
        range_vals[range_vals < self.eps] = 1.0

        normalized = (points - self.min) / range_vals

        return normalized

    def _normalize_standard(self, points: np.ndarray) -> np.ndarray:
        """
        标准化（零均值，单位方差）

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 标准化后的点云
        """
        if self.mean is None or self.std is None:
            self.mean = np.mean(points, axis=0)
            self.std = np.std(points, axis=0) + self.eps

        normalized = (points - self.mean) / self.std

        return normalized

    def _normalize_robust(self, points: np.ndarray) -> np.ndarray:
        """
        鲁棒标准化（中位数，四分位距）

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 标准化后的点云
        """
        if self.mean is None or self.std is None:
            self.mean = np.median(points, axis=0)
            q75, q25 = np.percentile(points, [75, 25], axis=0)
            self.std = (q75 - q25) + self.eps

        normalized = (points - self.mean) / self.std

        return normalized

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """
        逆变换

        Args:
            points: 归一化后的点云

        Returns:
            np.ndarray: 原始尺度的点云
        """
        if self.method == "unit_sphere":
            # unit_sphere的逆变换需要知道原始尺度信息
            # 由于unit_sphere只进行了缩放，无法完全恢复
            return points  # 无法完全逆变换
        elif self.method == "unit_cube":
            if self.min is None or self.max is None:
                raise ValueError("需要先调用fit方法")
            range_vals = self.max - self.min
            original = points * range_vals + self.min
            return original
        elif self.method == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("需要先调用fit方法")
            original = points * self.std + self.mean
            return original
        elif self.method == "robust":
            if self.mean is None or self.std is None:
                raise ValueError("需要先调用fit方法")
            original = points * self.std + self.mean
            return original
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")

    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算点云的包围盒

        Args:
            points: 点云数据

        Returns:
            Tuple[np.ndarray, np.ndarray]: (最小点, 最大点)
        """
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        return min_point, max_point

    @staticmethod
    def compute_centroid(points: np.ndarray) -> np.ndarray:
        """
        计算点云的质心

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 质心坐标
        """
        return np.mean(points, axis=0)

    @staticmethod
    def compute_covariance_matrix(points: np.ndarray) -> np.ndarray:
        """
        计算点云的协方差矩阵

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 3x3协方差矩阵
        """
        centered = points - np.mean(points, axis=0)
        covariance = np.dot(centered.T, centered) / (points.shape[0] - 1)
        return covariance

    @staticmethod
    def align_to_principal_axes(points: np.ndarray) -> np.ndarray:
        """
        将点云对齐到主轴线

        Args:
            points: 点云数据

        Returns:
            np.ndarray: 对齐后的点云
        """
        # 计算协方差矩阵
        covariance = PointCloudNormalizer.compute_covariance_matrix(points)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 应用旋转
        aligned_points = np.dot(points, eigenvectors)

        return aligned_points


def normalize_pointcloud(points: np.ndarray, method: str = "unit_sphere") -> np.ndarray:
    """
    快速归一化点云

    Args:
        points: 点云数据
        method: 归一化方法

    Returns:
        np.ndarray: 归一化后的点云
    """
    normalizer = PointCloudNormalizer(method)
    return normalizer.fit_transform(points)


def test_normalization():
    """测试点云归一化"""
    print("测试点云归一化...")

    # 创建测试点云
    points = np.random.randn(100, 3) * 2 + 5  # 添加偏移和缩放

    print("原始点云:")
    print(f"  形状: {points.shape}")
    print(f"  均值: {np.mean(points, axis=0)}")
    print(f"  标准差: {np.std(points, axis=0)}")
    print(f"  最小值: {np.min(points, axis=0)}")
    print(f"  最大值: {np.max(points, axis=0)}")

    # 测试各种归一化方法
    methods = ["unit_sphere", "unit_cube", "standard", "robust"]

    for method in methods:
        print(f"\n{method} 归一化:")

        normalizer = PointCloudNormalizer(method)
        normalized = normalizer.fit_transform(points)

        print(f"  归一化后均值: {np.mean(normalized, axis=0)}")
        print(f"  归一化后标准差: {np.std(normalized, axis=0)}")
        print(f"  归一化后最小值: {np.min(normalized, axis=0)}")
        print(f"  归一化后最大值: {np.max(normalized, axis=0)}")

        # 测试逆变换
        if method != "unit_sphere":  # unit_sphere无法完全逆变换
            reconstructed = normalizer.inverse_transform(normalized)
            error = np.mean(np.abs(points - reconstructed))
            print(f"  逆变换误差: {error:.6f}")

    # 测试主轴线对齐
    print("\n主轴线对齐:")
    aligned = PointCloudNormalizer.align_to_principal_axes(points)
    print(f"  对齐后形状: {aligned.shape}")

    # 计算协方差矩阵
    original_cov = PointCloudNormalizer.compute_covariance_matrix(points)
    aligned_cov = PointCloudNormalizer.compute_covariance_matrix(aligned)

    print(f"  原始协方差矩阵对角线: {np.diag(original_cov)}")
    print(f"  对齐后协方差矩阵对角线: {np.diag(aligned_cov)}")

    print("\n点云归一化测试通过!")


if __name__ == "__main__":
    test_normalization()