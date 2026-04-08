"""
Unit tests for training modules.
"""
import pytest
import torch
import tempfile
import os
from training.metrics import AccuracyMetric, ConfusionMatrixMetric
from training.loss_functions import ClassificationLoss
from training.callbacks import EarlyStopping, ModelCheckpoint
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


def test_accuracy_metric():
    """Test accuracy metric calculation."""
    metric = AccuracyMetric()

    # Test batch update
    predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])  # 3 samples, 2 classes
    labels = torch.tensor([0, 1, 0])

    metric.update(predictions, labels)
    accuracy = metric.compute()

    # predictions: sample0 class0 (correct), sample1 class1 (correct), sample2 class0 (correct)
    assert accuracy == 1.0  # 100% accuracy

    # Test reset
    metric.reset()
    assert metric.total == 0
    assert metric.correct == 0


def test_confusion_matrix():
    """Test confusion matrix metric."""
    num_classes = 3
    metric = ConfusionMatrixMetric(num_classes=num_classes)

    predictions = torch.tensor([
        [0.8, 0.1, 0.1],  # class 0
        [0.1, 0.7, 0.2],  # class 1
        [0.2, 0.3, 0.5],  # class 2 (predicted as class 2)
        [0.9, 0.05, 0.05],  # class 0
    ])
    labels = torch.tensor([0, 1, 2, 0])

    metric.update(predictions, labels)
    cm = metric.compute()

    assert cm.shape == (num_classes, num_classes)
    # Check diagonal (correct predictions) should be 4
    assert torch.trace(cm) == 4

    # Test classification report
    class_names = ["class_0", "class_1", "class_2"]
    report = metric.get_classification_report(class_names)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report


def test_loss_function():
    """Test loss function creation and computation."""
    # Create classification loss
    loss_fn = ClassificationLoss()

    # Test with sample data
    predictions = torch.randn(8, 10)  # batch_size=8, num_classes=10
    labels = torch.randint(0, 10, (8,))

    loss = loss_fn(predictions, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


def test_early_stopping():
    """Test early stopping callback."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        min_delta=0.01
    )

    # Simulate training with improving then worsening metrics
    metrics = [10.0, 9.0, 8.0, 8.5, 9.0, 9.5, 10.0]  # gets worse after epoch 3

    for epoch, metric_value in enumerate(metrics):
        should_stop = early_stopping.on_epoch_end(epoch, {"val_loss": metric_value})
        if epoch >= 5:  # patience=3, so should stop at epoch 6 (0-indexed)
            assert should_stop
            break

    # Test min mode (lower is better)
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    assert not early_stopping.on_epoch_end(0, {"val_loss": 10.0})
    assert not early_stopping.on_epoch_end(1, {"val_loss": 9.5})  # improving
    assert not early_stopping.on_epoch_end(2, {"val_loss": 9.7})  # slightly worse
    assert not early_stopping.on_epoch_end(3, {"val_loss": 9.8})  # worse again
    assert early_stopping.on_epoch_end(4, {"val_loss": 9.9})  # should stop


def test_model_checkpoint(tmp_path):
    """Test model checkpoint callback."""
    # Create a simple model
    model = torch.nn.Linear(10, 5)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "best_model.pth"

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    # Simulate improving accuracy
    checkpoint.on_epoch_end(0, {"val_accuracy": 0.8}, model=model)
    assert checkpoint_path.exists()

    # Worse accuracy - should not save
    checkpoint.on_epoch_end(1, {"val_accuracy": 0.7}, model=model)
    # Check that best metric is still 0.8
    assert checkpoint.best_metric == 0.8

    # Better accuracy - should save
    checkpoint.on_epoch_end(2, {"val_accuracy": 0.9}, model=model)
    assert checkpoint.best_metric == 0.9


def test_optimizer_creation():
    """Test optimizer creation."""
    model = torch.nn.Linear(10, 5)

    # Test Adam optimizer
    optimizer = create_optimizer(model, "adam", lr=0.001)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.001

    # Test AdamW optimizer
    optimizer = create_optimizer(model, "adamw", lr=0.01)
    assert isinstance(optimizer, torch.optim.AdamW)

    # Test SGD optimizer
    optimizer = create_optimizer(model, "sgd", lr=0.1, momentum=0.9)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9


def test_scheduler_creation():
    """Test scheduler creation."""
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test StepLR scheduler
    scheduler = create_scheduler(optimizer, "step", step_size=10, gamma=0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1

    # Test CosineAnnealingLR scheduler
    scheduler = create_scheduler(optimizer, "cosine", T_max=100)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    # Test ReduceLROnPlateau scheduler
    scheduler = create_scheduler(optimizer, "plateau", mode="min", patience=5)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_trainer_initialization():
    """Test trainer initialization."""
    from training.trainer import Trainer

    # Create simple model and data
    model = torch.nn.Linear(10, 5)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 10),
        torch.randint(0, 5, (100,))
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    # Create trainer
    from config.training_config import TrainingConfig
    config = TrainingConfig(
        epochs=2,
        learning_rate=0.001,
        batch_size=16
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    assert trainer.model == model
    assert trainer.config == config
    assert trainer.train_loader == train_loader
    assert trainer.val_loader == val_loader


if __name__ == "__main__":
    pytest.main([__file__, "-v"])