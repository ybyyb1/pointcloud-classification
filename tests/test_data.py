"""
Unit tests for data modules.
"""
import pytest
import torch
import numpy as np
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.pointcloud_sampling import random_sampling
from data.preprocessing.pointcloud_normalization import normalize_pointcloud


class MockDataset(BaseDataset):
    """Mock dataset for testing."""
    def __init__(self, num_samples=100, num_points=1024, num_classes=10):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self._data = [
            {
                "points": torch.randn(3, num_points),
                "label": torch.randint(0, num_classes, (1,)).item()
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self._data[idx]
        return {
            "points": item["points"].float(),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

    def get_statistics(self):
        """Get dataset statistics."""
        labels = [item["label"] for item in self._data]
        unique, counts = np.unique(labels, return_counts=True)
        return {
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "class_distribution": dict(zip([self.class_names[i] for i in unique], counts.tolist()))
        }


def test_base_dataset():
    """Test base dataset functionality."""
    dataset = MockDataset(num_samples=50, num_points=512, num_classes=5)
    assert len(dataset) == 50
    assert len(dataset.class_names) == 5

    # Test getitem
    sample = dataset[0]
    assert "points" in sample
    assert "label" in sample
    assert sample["points"].shape == (3, 512)
    assert isinstance(sample["label"], torch.Tensor)

    # Test statistics
    stats = dataset.get_statistics()
    assert stats["num_samples"] == 50
    assert stats["num_classes"] == 5


def test_pointcloud_sampling():
    """Test point cloud sampling."""
    # Create random point cloud
    points = torch.randn(3, 2048)  # 2048 points

    # Downsample to 1024 points
    sampled = random_sampling(points, 1024)
    assert sampled.shape == (3, 1024)

    # Upsample (should return original if target > source)
    sampled = random_sampling(points, 4096)
    assert sampled.shape == (3, 4096)

    # Test with numpy array
    points_np = np.random.randn(3, 2048)
    sampled_np = random_sampling(points_np, 1024)
    assert sampled_np.shape == (3, 1024)


def test_pointcloud_normalization():
    """Test point cloud normalization."""
    # Create random point cloud
    points = torch.randn(3, 1024)

    # Normalize
    normalized = normalize_pointcloud(points)

    # Check shape preserved
    assert normalized.shape == (3, 1024)

    # Check zero mean (approximately)
    mean = normalized.mean(dim=1, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    # Check unit sphere (approximately)
    max_dist = normalized.norm(dim=0).max()
    assert max_dist <= 1.0 + 1e-5


def test_data_augmentation():
    """Test data augmentation functions."""
    from data.preprocessing.pointcloud_augmentation import (
        random_rotation, random_translation, random_scaling
    )

    points = torch.randn(3, 1024)

    # Test random rotation
    rotated = random_rotation(points)
    assert rotated.shape == points.shape

    # Test random translation
    translated = random_translation(points, max_translation=0.1)
    assert translated.shape == points.shape

    # Test random scaling
    scaled = random_scaling(points, min_scale=0.8, max_scale=1.2)
    assert scaled.shape == points.shape


def test_dataloader_creation():
    """Test dataloader creation."""
    from torch.utils.data import DataLoader

    dataset = MockDataset(num_samples=100, num_points=1024, num_classes=10)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )

    # Iterate through dataloader
    batch = next(iter(dataloader))
    assert "points" in batch
    assert "label" in batch
    assert batch["points"].shape == (16, 3, 1024)
    assert batch["label"].shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])