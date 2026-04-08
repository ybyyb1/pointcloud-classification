"""
Unit tests for model implementations.
"""
import torch
import pytest
from models.model_factory import create_model
from models.point_transformer import PointTransformer
from models.pointnet import PointNet
from models.dgcnn import DGCNN


def test_model_factory():
    """Test model factory creation."""
    # Test Point Transformer
    model = create_model({
        "model_name": "point_transformer",
        "num_classes": 10,
        "num_points": 1024
    })
    assert isinstance(model, PointTransformer)
    assert model.num_classes == 10

    # Test PointNet
    model = create_model({
        "model_name": "pointnet",
        "num_classes": 15,
        "num_points": 1024
    })
    assert isinstance(model, PointNet)
    assert model.num_classes == 15

    # Test DGCNN
    model = create_model({
        "model_name": "dgcnn",
        "num_classes": 20,
        "num_points": 1024
    })
    assert isinstance(model, DGCNN)
    assert model.num_classes == 20


def test_point_transformer_forward(sample_point_cloud):
    """Test PointTransformer forward pass."""
    model = PointTransformer(num_classes=15, num_points=1024)
    output = model(sample_point_cloud)
    assert output.shape == (1, 15)  # batch, classes


def test_pointnet_forward(sample_point_cloud):
    """Test PointNet forward pass."""
    model = PointNet(num_classes=15, num_points=1024)
    output = model(sample_point_cloud)
    assert output.shape == (1, 15)


def test_dgcnn_forward(sample_point_cloud):
    """Test DGCNN forward pass."""
    model = DGCNN(num_classes=15, num_points=1024)
    output = model(sample_point_cloud)
    assert output.shape == (1, 15)


def test_model_parameters():
    """Test model parameter counting."""
    model = PointTransformer(num_classes=15, num_points=1024)
    total_params, trainable_params = model.count_parameters()
    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    model = PointTransformer(num_classes=15, num_points=1024)

    # Save model
    save_path = tmp_path / "test_model.pth"
    model.save(save_path)
    assert save_path.exists()

    # Load model
    from models.model_factory import load_model
    loaded_model = load_model(save_path)
    assert isinstance(loaded_model, PointTransformer)
    assert loaded_model.num_classes == 15

    # Test forward pass consistency
    input_tensor = torch.randn(1, 3, 1024)
    with torch.no_grad():
        orig_output = model(input_tensor)
        loaded_output = loaded_model(input_tensor)

    torch.testing.assert_close(orig_output, loaded_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])