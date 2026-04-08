"""
Pytest configuration for point cloud classification tests.
"""
import pytest
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


@pytest.fixture
def sample_point_cloud():
    """Generate a sample point cloud for testing."""
    import torch
    return torch.randn(1, 3, 1024)  # (batch, channels, points)


@pytest.fixture
def sample_batch():
    """Generate a sample batch for testing."""
    import torch
    batch_size = 4
    num_points = 1024
    num_channels = 3
    points = torch.randn(batch_size, num_channels, num_points)
    labels = torch.randint(0, 15, (batch_size,))
    return points, labels


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from config.base_config import SystemConfig
    return SystemConfig()