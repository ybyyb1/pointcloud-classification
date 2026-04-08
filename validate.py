#!/usr/bin/env python3
"""
Validation script for Point Cloud Classification System.
Checks that all modules can be imported and basic functionality works.
"""
import importlib
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# List of modules to test
MODULES_TO_TEST = [
    # Config modules
    ("config.base_config", ["SystemConfig", "DatasetConfig", "ModelConfig", "TrainingConfig"]),
    ("config.dataset_config", ["SCANOBJECTNN_CONFIG", "get_dataset_config"]),
    ("config.model_config", ["POINT_TRANSFORMER_CONFIG", "get_model_config"]),
    ("config.training_config", ["DEFAULT_TRAINING_CONFIG", "get_training_config"]),

    # Data modules
    ("data.datasets.base_dataset", ["BaseDataset"]),
    ("data.datasets.scanobjectnn_dataset", ["ScanObjectNNDataset"]),
    ("data.datasets.s3dis_dataset", ["S3DISDataset"]),
    ("data.preprocessing.pointcloud_sampling", ["random_sampling"]),
    ("data.preprocessing.pointcloud_normalization", ["normalize_pointcloud"]),

    # Model modules
    ("models.base_model", ["BaseModel"]),
    ("models.point_transformer", ["PointTransformer"]),
    ("models.pointnet", ["PointNet"]),
    ("models.dgcnn", ["DGCNN"]),
    ("models.model_factory", ["ModelFactory", "create_model"]),

    # Training modules
    ("training.trainer", ["Trainer"]),
    ("training.metrics", ["AccuracyMetric", "ConfusionMatrixMetric"]),
    ("training.callbacks", ["EarlyStopping", "ModelCheckpoint"]),
    ("training.optimizer", ["create_optimizer"]),
    ("training.scheduler", ["create_scheduler"]),
    ("training.loss_functions", ["ClassificationLoss"]),

    # Utility modules
    ("utils.logger", ["setup_logger", "ProgressLogger"]),

    # Evaluation modules
    ("evaluation.analyzer", ["ResultAnalyzer"]),
]


def test_module_import(module_name, expected_attributes=None):
    """Test importing a module and its attributes."""
    try:
        module = importlib.import_module(module_name)

        # Check attributes if specified
        if expected_attributes:
            missing_attrs = []
            for attr in expected_attributes:
                if not hasattr(module, attr):
                    missing_attrs.append(attr)

            if missing_attrs:
                return False, f"Missing attributes: {missing_attrs}"

        return True, "OK"

    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def test_basic_functionality():
    """Test basic functionality of key components."""
    tests = []

    # Test 1: Config creation
    try:
        from config.base_config import SystemConfig
        config = SystemConfig()
        tests.append(("Config creation", True, "OK"))
    except Exception as e:
        tests.append(("Config creation", False, str(e)))

    # Test 2: Model factory
    try:
        from models.model_factory import create_model
        # Just test that function exists
        tests.append(("Model factory", True, "OK"))
    except Exception as e:
        tests.append(("Model factory", False, str(e)))

    # Test 3: Data preprocessing
    try:
        import torch
        from data.preprocessing.pointcloud_sampling import random_sampling

        points = torch.randn(3, 2048)
        sampled = random_sampling(points, 1024)
        assert sampled.shape == (3, 1024)
        tests.append(("Point cloud sampling", True, "OK"))
    except Exception as e:
        tests.append(("Point cloud sampling", False, str(e)))

    # Test 4: Training metrics
    try:
        from training.metrics import AccuracyMetric
        metric = AccuracyMetric()
        tests.append(("Training metrics", True, "OK"))
    except Exception as e:
        tests.append(("Training metrics", False, str(e)))

    # Test 5: Logger
    try:
        from utils.logger import setup_logger
        # Just test import
        tests.append(("Logger", True, "OK"))
    except Exception as e:
        tests.append(("Logger", False, str(e)))

    return tests


def main():
    print("=" * 70)
    print("Point Cloud Classification System - Validation")
    print("=" * 70)

    # Test module imports
    print("\n1. Testing module imports...")
    import_results = []

    for module_name, expected_attrs in MODULES_TO_TEST:
        success, message = test_module_import(module_name, expected_attrs)
        status = "+" if success else "X"
        import_results.append((module_name, success, message))
        print(f"  {status} {module_name}")

        if not success and message != "OK":
            print(f"    -> {message}")

    # Test basic functionality
    print("\n2. Testing basic functionality...")
    func_results = test_basic_functionality()

    for test_name, success, message in func_results:
        status = "+" if success else "X"
        print(f"  {status} {test_name}")
        if not success:
            print(f"    -> {message}")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_imports = len(import_results)
    successful_imports = sum(1 for _, success, _ in import_results if success)
    import_success_rate = successful_imports / total_imports * 100

    total_func = len(func_results)
    successful_func = sum(1 for _, success, _ in func_results if success)
    func_success_rate = successful_func / total_func * 100

    print(f"Module Imports: {successful_imports}/{total_imports} ({import_success_rate:.1f}%)")
    print(f"Functionality Tests: {successful_func}/{total_func} ({func_success_rate:.1f}%)")

    # List failed imports
    failed_imports = [(name, msg) for name, success, msg in import_results if not success]
    if failed_imports:
        print("\nFailed Imports:")
        for name, msg in failed_imports:
            print(f"  - {name}: {msg}")

    # List failed functionality tests
    failed_func = [(name, msg) for name, success, msg in func_results if not success]
    if failed_func:
        print("\nFailed Functionality Tests:")
        for name, msg in failed_func:
            print(f"  - {name}: {msg}")

    # Overall status
    all_success = (successful_imports == total_imports and successful_func == total_func)

    print("\n" + "=" * 70)
    if all_success:
        print("[PASS] VALIDATION PASSED - All tests successful!")
    else:
        print("[WARN] VALIDATION HAS ISSUES - Some tests failed")
        print("\nRecommendations:")
        if failed_imports:
            print("  - Check missing dependencies in requirements.txt")
            print("  - Verify module paths and __init__.py files")
        if failed_func:
            print("  - Check implementation of failed functionality")
            print("  - Verify data types and shapes")

    print("\nNext steps:")
    print("  1. Install missing dependencies: pip install -r requirements.txt")
    print("  2. Run full test suite: python -m pytest tests/ -v")
    print("  3. Try example commands: python main.py --help")
    print("  4. Run the web interface: streamlit run app.py")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())