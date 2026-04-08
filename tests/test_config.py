"""
Unit tests for configuration modules.
"""
import pytest
import tempfile
import yaml
import json
import os
from config import (
    SystemConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    DatasetType,
    ModelType,
    load_config,
    save_config
)


def test_system_config():
    """Test system configuration."""
    config = SystemConfig()

    # Check default values
    assert config.project_name == "Point Cloud Classification System"
    assert config.version == "1.0.0"

    # Check nested configs
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)

    # Test to_dict and from_dict
    config_dict = config.to_dict()
    assert "project_name" in config_dict
    assert "dataset" in config_dict
    assert "model" in config_dict
    assert "training" in config_dict

    # Create from dict
    new_config = SystemConfig.from_dict(config_dict)
    assert new_config.project_name == config.project_name


def test_dataset_config():
    """Test dataset configuration."""
    # Test default config
    config = DatasetConfig()
    assert config.dataset_type == DatasetType.SCANOBJECTNN
    assert config.num_points == 1024
    assert config.batch_size == 32

    # Test custom config
    config = DatasetConfig(
        dataset_type=DatasetType.S3DIS,
        data_dir="./data/s3dis",
        num_points=2048,
        batch_size=64,
        num_workers=4
    )
    assert config.dataset_type == DatasetType.S3DIS
    assert config.data_dir == "./data/s3dis"
    assert config.num_points == 2048
    assert config.batch_size == 64
    assert config.num_workers == 4

    # Test enum values
    assert DatasetType.SCANOBJECTNN.value == "scanobjectnn"
    assert DatasetType.S3DIS.value == "s3dis"
    assert DatasetType.CUSTOM.value == "custom"


def test_model_config():
    """Test model configuration."""
    # Test default config
    config = ModelConfig()
    assert config.model_type == ModelType.POINT_TRANSFORMER
    assert config.num_classes == 15

    # Test custom config
    config = ModelConfig(
        model_type=ModelType.POINTNET,
        num_classes=10,
        point_transformer_dim=256,
        point_transformer_depth=4
    )
    assert config.model_type == ModelType.POINTNET
    assert config.num_classes == 10
    assert config.point_transformer_dim == 256
    assert config.point_transformer_depth == 4

    # Test enum values
    assert ModelType.POINT_TRANSFORMER.value == "point_transformer"
    assert ModelType.POINTNET.value == "pointnet"
    assert ModelType.POINTNET2.value == "pointnet2"
    assert ModelType.DGCNN.value == "dgcnn"


def test_training_config():
    """Test training configuration."""
    # Test default config
    config = TrainingConfig()
    assert config.epochs == 300
    assert config.learning_rate == 0.001
    assert config.batch_size == 32
    assert config.optimizer == "adamw"

    # Test custom config
    config = TrainingConfig(
        epochs=100,
        learning_rate=0.01,
        batch_size=64,
        optimizer="sgd",
        scheduler="cosine",
        early_stopping_patience=20,
        checkpoint_dir="./my_checkpoints"
    )
    assert config.epochs == 100
    assert config.learning_rate == 0.01
    assert config.batch_size == 64
    assert config.optimizer == "sgd"
    assert config.scheduler == "cosine"
    assert config.early_stopping_patience == 20
    assert config.checkpoint_dir == "./my_checkpoints"


def test_config_serialization_yaml(tmp_path):
    """Test configuration serialization to YAML."""
    config = SystemConfig()
    config.project_name = "Test Project"
    config.version = "2.0.0"

    # Save to YAML
    yaml_file = tmp_path / "test_config.yaml"
    save_config(config, yaml_file)

    # Check file exists
    assert yaml_file.exists()

    # Load from YAML
    loaded_config = load_config(yaml_file)
    assert isinstance(loaded_config, SystemConfig)
    assert loaded_config.project_name == "Test Project"
    assert loaded_config.version == "2.0.0"


def test_config_serialization_json(tmp_path):
    """Test configuration serialization to JSON."""
    config = SystemConfig()
    config.project_name = "JSON Test"
    config.version = "3.0.0"

    # Save to JSON
    json_file = tmp_path / "test_config.json"
    save_config(config, json_file)

    # Check file exists
    assert json_file.exists()

    # Load from JSON
    loaded_config = load_config(json_file)
    assert isinstance(loaded_config, SystemConfig)
    assert loaded_config.project_name == "JSON Test"
    assert loaded_config.version == "3.0.0"


def test_dataset_config_helpers():
    """Test dataset configuration helper functions."""
    from config.dataset_config import (
        get_dataset_config,
        get_class_names,
        SCANOBJECTNN_CLASSES
    )

    # Test get_dataset_config
    scanobjectnn_config = get_dataset_config(DatasetType.SCANOBJECTNN)
    assert "data_dir" in scanobjectnn_config
    assert "num_points" in scanobjectnn_config

    s3dis_config = get_dataset_config(DatasetType.S3DIS)
    assert "data_dir" in s3dis_config
    assert "num_points" in s3dis_config

    # Test get_class_names
    scanobjectnn_classes = get_class_names(DatasetType.SCANOBJECTNN)
    assert len(scanobjectnn_classes) > 0
    assert isinstance(scanobjectnn_classes, list)

    # Test SCANOBJECTNN_CLASSES
    assert len(SCANOBJECTNN_CLASSES) > 0
    assert "chair" in SCANOBJECTNN_CLASSES or "chair" in [c.lower() for c in SCANOBJECTNN_CLASSES]


def test_model_config_helpers():
    """Test model configuration helper functions."""
    from config.model_config import (
        get_model_config,
        get_model_parameters,
        POINT_TRANSFORMER_CONFIG
    )

    # Test get_model_config
    pt_config = get_model_config(ModelType.POINT_TRANSFORMER)
    assert "num_classes" in pt_config
    assert "point_transformer_dim" in pt_config

    pn_config = get_model_config(ModelType.POINTNET)
    assert "num_classes" in pn_config

    # Test get_model_parameters
    pt_params = get_model_parameters(ModelType.POINT_TRANSFORMER, num_classes=10, num_points=1024)
    assert "num_classes" in pt_params
    assert pt_params["num_classes"] == 10
    assert "num_points" in pt_params
    assert pt_params["num_points"] == 1024

    # Test POINT_TRANSFORMER_CONFIG
    assert "point_transformer_dim" in POINT_TRANSFORMER_CONFIG
    assert "point_transformer_depth" in POINT_TRANSFORMER_CONFIG


def test_training_config_helpers():
    """Test training configuration helper functions."""
    from config.training_config import (
        get_training_config,
        create_optimizer_config,
        create_scheduler_config,
        estimate_training_time,
        DEFAULT_TRAINING_CONFIG
    )

    # Test get_training_config
    default_config = get_training_config("default")
    assert "epochs" in default_config
    assert "learning_rate" in default_config

    kaggle_config = get_training_config("kaggle")
    assert "epochs" in kaggle_config

    # Test create_optimizer_config
    optimizer_config = create_optimizer_config("adamw", lr=0.001)
    assert optimizer_config["type"] == "adamw"
    assert optimizer_config["lr"] == 0.001

    # Test create_scheduler_config
    scheduler_config = create_scheduler_config("cosine", T_max=100)
    assert scheduler_config["type"] == "cosine"
    assert scheduler_config["T_max"] == 100

    # Test estimate_training_time
    time_estimate = estimate_training_time(
        epochs=100,
        dataset_size=1000,
        batch_size=32,
        use_gpu=True
    )
    assert isinstance(time_estimate, str)
    assert "小时" in time_estimate or "minutes" in time_estimate or "seconds" in time_estimate

    # Test DEFAULT_TRAINING_CONFIG
    assert "epochs" in DEFAULT_TRAINING_CONFIG
    assert "learning_rate" in DEFAULT_TRAINING_CONFIG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])