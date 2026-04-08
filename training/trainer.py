"""
训练器
模型训练的主要类
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

from models.base_model import BaseModel
from config import TrainingConfig
from .metrics import AccuracyMetric, AverageMetric
from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """基础训练器"""

    def __init__(self, model: BaseModel, config: TrainingConfig,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 test_loader: Optional[torch.utils.data.DataLoader] = None):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            test_loader: 测试数据加载器（可选）
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 训练状态
        self.epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }

        # 回调函数
        self.callbacks = []

        # 创建优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 创建损失函数
        self.criterion = self._create_loss_function()

        # 创建指标
        self.train_metrics = {
            "loss": AverageMetric(),
            "accuracy": AccuracyMetric()
        }
        self.val_metrics = {
            "loss": AverageMetric(),
            "accuracy": AccuracyMetric()
        }

        # 检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        print(f"训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  模型: {model.model_name}")
        print(f"  优化器: {self.optimizer.__class__.__name__}")
        print(f"  损失函数: {self.criterion.__class__.__name__}")
        print(f"  训练样本: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  验证样本: {len(val_loader.dataset)}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.config.optimizer.lower()

        if optimizer_name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        return optimizer

    def _create_scheduler(self) -> Optional[Any]:
        """创建学习率调度器"""
        scheduler_name = self.config.scheduler.lower()

        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.cosine_t_max,
                eta_min=1e-6
            )
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_milestones[0],
                gamma=self.config.step_gamma
            )
        elif scheduler_name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                min_lr=1e-6
            )
        else:
            scheduler = None

        return scheduler

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        loss_name = self.config.loss_function.lower()

        if loss_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss_name == "label_smoothing":
            criterion = LabelSmoothingCrossEntropy(smoothing=self.config.label_smoothing)
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

        return criterion

    def add_callback(self, callback: Any) -> None:
        """添加回调函数"""
        self.callbacks.append(callback)

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics["loss"].reset()
        self.train_metrics["accuracy"].reset()

        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {self.epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(points)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )

            # 参数更新
            self.optimizer.step()

            # 计算指标
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean()

                self.train_metrics["loss"].update(loss.item(), points.size(0))
                self.train_metrics["accuracy"].update(acc.item(), points.size(0))

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{self.train_metrics['loss'].avg:.4f}",
                "acc": f"{self.train_metrics['accuracy'].avg:.4f}"
            })

        # 计算epoch指标
        epoch_loss = self.train_metrics["loss"].avg
        epoch_acc = self.train_metrics["accuracy"].avg

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def validate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        self.val_metrics["loss"].reset()
        self.val_metrics["accuracy"].reset()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="验证"):
                # 准备数据
                points = batch["points"].to(self.device)
                labels = batch["label"].to(self.device)

                # 前向传播
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

                # 计算指标
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean()

                self.val_metrics["loss"].update(loss.item(), points.size(0))
                self.val_metrics["accuracy"].update(acc.item(), points.size(0))

                # 收集预测结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算验证指标
        val_loss = self.val_metrics["loss"].avg
        val_acc = self.val_metrics["accuracy"].avg

        return {
            "loss": val_loss,
            "accuracy": val_acc,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels)
        }

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            num_epochs: 训练轮数，如果为None则使用配置中的值

        Returns:
            Dict[str, List[float]]: 训练历史
        """
        if num_epochs is None:
            num_epochs = self.config.epochs

        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"检查点保存目录: {self.config.checkpoint_dir}")

        # 训练循环
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # 训练一个epoch
            train_results = self.train_epoch()
            train_loss = train_results["loss"]
            train_acc = train_results["accuracy"]

            # 验证
            val_results = None
            if self.val_loader:
                val_results = self.validate(self.val_loader)
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
            else:
                val_loss = 0.0
                val_acc = 0.0

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            # 计算epoch时间
            epoch_time = time.time() - start_time

            # 打印epoch结果
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            if val_results:
                print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"  学习率: {current_lr:.6f}")

            # 保存最佳模型
            if val_results and val_acc > self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(f"best_model.pth", val_acc)
                print(f"  最佳模型已保存 (准确率: {val_acc:.4f})")

            # 定期保存检查点
            if (epoch + 1) % self.config.save_checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", val_acc)

            # 执行回调函数
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, train_results, val_results)

            # 早停检查
            if hasattr(self.config, 'early_stopping_patience') and self.config.early_stopping_patience > 0:
                if epoch - self.best_epoch >= self.config.early_stopping_patience:
                    print(f"\n早停: {self.config.early_stopping_patience} 个epoch未改善")
                    break

        print(f"\n训练完成!")
        print(f"  最佳验证准确率: {self.best_metric:.4f} (Epoch {self.best_epoch + 1})")

        return self.history

    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """评估模型"""
        print("评估模型...")
        results = self.validate(loader)

        print(f"评估结果:")
        print(f"  损失: {results['loss']:.4f}")
        print(f"  准确率: {results['accuracy']:.4f}")

        return results

    def save_checkpoint(self, filename: str, metric: float = 0.0) -> None:
        """保存检查点"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)

        self.model.save(
            checkpoint_path,
            save_optimizer=True,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            metrics={"val_accuracy": metric}
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        loaded_info = self.model.load(
            checkpoint_path,
            load_optimizer=True,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )

        self.epoch = loaded_info.get("epoch", 0)
        self.best_metric = loaded_info.get("metrics", {}).get("val_accuracy", 0.0)

        print(f"检查点加载完成: epoch={self.epoch}, 准确率={self.best_metric:.4f}")

    def save_history(self, filepath: str) -> None:
        """保存训练历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"训练历史保存到: {filepath}")

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.history["train_loss"]) + 1)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 损失曲线
            axes[0, 0].plot(epochs, self.history["train_loss"], 'b-', label='训练损失')
            if self.val_loader:
                axes[0, 0].plot(epochs, self.history["val_loss"], 'r-', label='验证损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].set_title('训练和验证损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 准确率曲线
            axes[0, 1].plot(epochs, self.history["train_accuracy"], 'b-', label='训练准确率')
            if self.val_loader:
                axes[0, 1].plot(epochs, self.history["val_accuracy"], 'r-', label='验证准确率')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('准确率')
            axes[0, 1].set_title('训练和验证准确率')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 学习率曲线
            axes[1, 0].plot(epochs, self.history["learning_rate"], 'g-')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].grid(True)

            # 最佳epoch标记
            axes[1, 1].axis('off')
            axes[1, 1].text(0.1, 0.8, f'最佳验证准确率: {self.best_metric:.4f}', fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'最佳Epoch: {self.best_epoch + 1}', fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'总Epoch数: {len(epochs)}', fontsize=12)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"训练历史图表保存到: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("警告: matplotlib未安装，无法绘制图表")


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MixedPrecisionTrainer(Trainer):
    """混合精度训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

        print(f"混合精度训练启用: {self.config.use_amp}")

    def train_epoch(self) -> Dict[str, float]:
        """混合精度训练一个epoch"""
        self.model.train()
        self.train_metrics["loss"].reset()
        self.train_metrics["accuracy"].reset()

        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {self.epoch + 1} (混合精度)")
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device)

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

            # 反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            if self.config.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )

            # 参数更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # 计算指标
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean()

                self.train_metrics["loss"].update(loss.item(), points.size(0))
                self.train_metrics["accuracy"].update(acc.item(), points.size(0))

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{self.train_metrics['loss'].avg:.4f}",
                "acc": f"{self.train_metrics['accuracy'].avg:.4f}"
            })

        # 计算epoch指标
        epoch_loss = self.train_metrics["loss"].avg
        epoch_acc = self.train_metrics["accuracy"].avg

        return {"loss": epoch_loss, "accuracy": epoch_acc}


class KaggleTrainer(MixedPrecisionTrainer):
    """Kaggle优化训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Kaggle特定设置
        if self.config.kaggle_gpu_memory_limit is not None:
            torch.cuda.set_per_process_memory_fraction(
                self.config.kaggle_gpu_memory_limit / 16.0  # P100有16GB
            )

        print(f"Kaggle训练器初始化完成")
        print(f"  GPU内存限制: {self.config.kaggle_gpu_memory_limit} GB")


def test_trainer():
    """测试训练器"""
    print("测试训练器...")

    import tempfile
    import shutil

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")

    try:
        # 创建测试数据
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # 创建虚拟数据集
        num_samples = 100
        num_points = 1024
        num_classes = 5

        points = torch.randn(num_samples, num_points, 3)
        labels = torch.randint(0, num_classes, (num_samples,))

        dataset = TensorDataset(points, labels)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 创建模型
        from models.pointnet import PointNet
        model = PointNet(num_classes=num_classes, use_tnet=False)

        # 创建配置
        from config import TrainingConfig
        config = TrainingConfig(
            epochs=2,  # 只训练2个epoch用于测试
            learning_rate=0.001,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            save_checkpoint_interval=1
        )

        # 创建训练器
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # 训练
        history = trainer.train()

        print(f"\n训练历史:")
        print(f"  训练损失: {history['train_loss']}")
        print(f"  训练准确率: {history['train_accuracy']}")
        print(f"  验证损失: {history['val_loss']}")
        print(f"  验证准确率: {history['val_accuracy']}")

        # 评估
        eval_results = trainer.evaluate(val_loader)
        print(f"\n评估结果: {eval_results}")

        # 保存历史
        history_file = os.path.join(temp_dir, "history.json")
        trainer.save_history(history_file)

        # 测试混合精度训练器
        print("\n测试混合精度训练器...")
        mixed_trainer = MixedPrecisionTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # 测试Kaggle训练器
        print("\n测试Kaggle训练器...")
        kaggle_config = TrainingConfig(
            epochs=1,
            learning_rate=0.001,
            checkpoint_dir=os.path.join(temp_dir, "kaggle_checkpoints"),
            use_amp=True,
            kaggle_gpu_memory_limit=14.0
        )

        kaggle_trainer = KaggleTrainer(
            model=model,
            config=kaggle_config,
            train_loader=train_loader,
            val_loader=val_loader
        )

        print("训练器测试通过!")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"临时目录已清理")


if __name__ == "__main__":
    test_trainer()