"""
训练回调函数模块
包含各种训练回调函数
"""

import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime


class Callback:
    """回调函数基类"""

    def __init__(self):
        pass

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时调用"""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时调用"""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """batch开始时调用"""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """batch结束时调用"""
        pass


class EarlyStopping(Callback):
    """早停回调函数"""

    def __init__(self, monitor: str = "val_accuracy", patience: int = 10,
                 mode: str = "max", min_delta: float = 0.0):
        """
        初始化早停回调

        Args:
            monitor: 监控的指标名称
            patience: 容忍轮数
            mode: "min" 或 "max"
            min_delta: 最小变化量
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时重置状态"""
        self.best_value = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时检查是否早停"""
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        # 检查是否改善
        if self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            print(f"早停: 指标改善到 {current_value:.4f} (最佳: {self.best_value:.4f})")
        else:
            self.wait += 1
            print(f"早停: 等待 {self.wait}/{self.patience} (最佳: {self.best_value:.4f})")

        # 检查是否早停
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            print(f"早停触发: {self.patience} 个epoch未改善")


class ModelCheckpoint(Callback):
    """模型检查点回调函数"""

    def __init__(self, filepath: str, monitor: str = "val_accuracy",
                 mode: str = "max", save_best_only: bool = True,
                 save_weights_only: bool = False, save_freq: int = 1):
        """
        初始化模型检查点

        Args:
            filepath: 文件路径，可以包含格式化字符串如 "model_{epoch:02d}.pth"
            monitor: 监控的指标名称
            mode: "min" 或 "max"
            save_best_only: 是否只保存最佳模型
            save_weights_only: 是否只保存权重
            save_freq: 保存频率（epoch）
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq

        self.best_value = -np.inf if mode == "max" else np.inf
        self.best_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时重置状态"""
        self.best_value = -np.inf if self.mode == "max" else np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时保存模型"""
        if logs is None:
            return

        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        # 检查是否需要保存
        should_save = False

        if self.save_best_only:
            if self.mode == "max":
                improved = current_value > self.best_value
            else:
                improved = current_value < self.best_value

            if improved:
                self.best_value = current_value
                self.best_epoch = epoch
                should_save = True
                print(f"模型检查点: 新最佳指标 {current_value:.4f}")
        else:
            # 定期保存
            if (epoch + 1) % self.save_freq == 0:
                should_save = True

        if should_save:
            # 生成文件路径
            if "{" in self.filepath:
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                filepath = self.filepath

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 保存模型
            from models.base_model import BaseModel
            model = logs.get("model")
            if model is not None and isinstance(model, BaseModel):
                save_info = {
                    "epoch": epoch,
                    "metrics": {self.monitor: current_value},
                    "logs": logs
                }
                model.save(filepath, **save_info)
                print(f"模型保存到: {filepath}")


class TensorBoardLogger(Callback):
    """TensorBoard日志回调函数"""

    def __init__(self, log_dir: str = "./logs", update_freq: str = "epoch"):
        """
        初始化TensorBoard日志

        Args:
            log_dir: 日志目录
            update_freq: 更新频率，"epoch" 或 "batch"
        """
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq

        self.writer = None
        self.batch_count = 0
        self.epoch_count = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时创建TensorBoard写入器"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, timestamp)
            self.writer = SummaryWriter(log_path)
            print(f"TensorBoard日志目录: {log_path}")
        except ImportError:
            print("警告: tensorboard未安装，跳过TensorBoard日志")
            self.writer = None

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时关闭TensorBoard写入器"""
        if self.writer is not None:
            self.writer.close()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时记录日志"""
        if self.writer is None or logs is None:
            return

        self.epoch_count += 1

        # 记录标量指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, self.epoch_count)

        # 记录学习率
        optimizer = logs.get("optimizer")
        if optimizer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group.get("lr")
                if lr is not None:
                    self.writer.add_scalar(f"learning_rate/group_{i}", lr, self.epoch_count)

        # 记录模型图（只在第一次epoch）
        if self.epoch_count == 1:
            model = logs.get("model")
            sample_input = logs.get("sample_input")
            if model is not None and sample_input is not None:
                try:
                    self.writer.add_graph(model, sample_input)
                except Exception as e:
                    print(f"无法添加模型图到TensorBoard: {e}")

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """batch结束时记录日志"""
        if self.writer is None or logs is None or self.update_freq != "batch":
            return

        self.batch_count += 1

        # 记录batch指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"batch_{key}", value, self.batch_count)


class ProgressLogger(Callback):
    """进度日志回调函数"""

    def __init__(self, log_file: str = "./logs/training.log", verbose: int = 1):
        """
        初始化进度日志

        Args:
            log_file: 日志文件路径
            verbose: 详细程度，0=不输出，1=输出到控制台和文件，2=只输出到文件
        """
        super().__init__()
        self.log_file = log_file
        self.verbose = verbose
        self.log_buffer = []

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时记录开始时间"""
        self.start_time = time.time()
        message = f"\n{'='*60}\n训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}"

        if self.verbose >= 1:
            print(message)

        self._write_to_file(message)

        # 记录训练参数
        if logs is not None:
            params_message = "\n训练参数:\n"
            for key, value in logs.items():
                params_message += f"  {key}: {value}\n"

            if self.verbose >= 2:
                print(params_message)
            self._write_to_file(params_message)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时记录总时间"""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        message = f"\n{'='*60}\n训练结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"总时间: {hours:02d}:{minutes:02d}:{seconds:02d}\n{'='*60}"

        if self.verbose >= 1:
            print(message)

        self._write_to_file(message)

        # 刷新缓冲区到文件
        self._flush_buffer()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时记录进度"""
        if logs is None:
            return

        # 构建消息
        message = f"\nEpoch {epoch + 1}:\n"
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                message += f"  {key}: {value:.4f}\n"
            else:
                message += f"  {key}: {value}\n"

        # 添加时间信息
        elapsed_time = time.time() - self.start_time
        message += f"  已用时间: {elapsed_time:.1f}s\n"

        if self.verbose >= 1:
            print(message)

        self._write_to_file(message)

    def _write_to_file(self, message: str):
        """写入文件（带缓冲）"""
        self.log_buffer.append(message)

        # 每10条消息刷新一次缓冲区
        if len(self.log_buffer) >= 10:
            self._flush_buffer()

    def _flush_buffer(self):
        """刷新缓冲区到文件"""
        if not self.log_buffer:
            return

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for message in self.log_buffer:
                    f.write(message + '\n')
            self.log_buffer.clear()
        except Exception as e:
            print(f"写入日志文件失败: {e}")


class LearningRateSchedulerCallback(Callback):
    """学习率调度器回调函数"""

    def __init__(self, scheduler):
        """
        初始化学习率调度器回调

        Args:
            scheduler: 学习率调度器
        """
        super().__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时更新学习率"""
        if logs is None:
            return

        # 根据监控指标更新调度器
        if hasattr(self.scheduler, 'step'):
            # 如果是ReduceLROnPlateau，需要指标
            if hasattr(self.scheduler, 'mode'):
                monitor_value = logs.get('val_accuracy')
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
            else:
                self.scheduler.step()

        # 记录当前学习率
        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else \
                     self.scheduler.optimizer.param_groups[0]['lr']
        logs['learning_rate'] = current_lr


class CSVLogger(Callback):
    """CSV日志回调函数"""

    def __init__(self, filename: str = "./logs/training.csv", separator: str = ",", append: bool = False):
        """
        初始化CSV日志

        Args:
            filename: CSV文件路径
            separator: 分隔符
            append: 是否追加到现有文件
        """
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys = None

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时创建CSV文件"""
        if not self.append and os.path.exists(self.filename):
            os.remove(self.filename)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """epoch结束时记录到CSV"""
        if logs is None:
            return

        # 确定列名
        if self.keys is None:
            self.keys = ['epoch'] + list(logs.keys())

        # 写入文件
        file_exists = os.path.exists(self.filename)

        with open(self.filename, 'a', encoding='utf-8') as f:
            # 写入表头
            if not file_exists or not self.append:
                f.write(self.separator.join(self.keys) + '\n')

            # 写入数据
            row = [str(epoch)]
            for key in self.keys[1:]:
                value = logs.get(key, '')
                if isinstance(value, (int, float)):
                    row.append(f"{value:.6f}")
                else:
                    row.append(str(value))

            f.write(self.separator.join(row) + '\n')


def create_callback(callback_type: str, **kwargs) -> Callback:
    """
    创建回调函数

    Args:
        callback_type: 回调函数类型
        **kwargs: 回调函数参数

    Returns:
        Callback: 回调函数对象
    """
    callback_map = {
        "early_stopping": EarlyStopping,
        "model_checkpoint": ModelCheckpoint,
        "tensorboard": TensorBoardLogger,
        "progress_logger": ProgressLogger,
        "csv_logger": CSVLogger,
    }

    if callback_type not in callback_map:
        raise ValueError(f"未知的回调函数类型: {callback_type}")

    return callback_map[callback_type](**kwargs)


def test_callbacks():
    """测试回调函数模块"""
    print("测试回调函数模块...")

    # 测试EarlyStopping
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=3, mode="max")
    early_stopping.on_train_begin()

    # 模拟一些epoch
    logs_list = [
        {"val_accuracy": 0.7},
        {"val_accuracy": 0.75},
        {"val_accuracy": 0.72},
        {"val_accuracy": 0.71},
        {"val_accuracy": 0.70},
    ]

    for epoch, logs in enumerate(logs_list):
        early_stopping.on_epoch_end(epoch, logs)
        if early_stopping.wait >= early_stopping.patience:
            print(f"早停在epoch {epoch}触发")
            break

    print(f"最佳指标: {early_stopping.best_value:.4f} (epoch {early_stopping.best_epoch})")

    # 测试ModelCheckpoint（模拟）
    model_checkpoint = ModelCheckpoint(
        filepath="./test_checkpoints/model_{epoch:02d}.pth",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    print(f"ModelCheckpoint初始化成功")

    # 测试ProgressLogger
    progress_logger = ProgressLogger(log_file="./test_logs/progress.log", verbose=1)
    progress_logger.on_train_begin({"epochs": 10, "learning_rate": 0.001})

    for epoch in range(3):
        progress_logger.on_epoch_end(epoch, {
            "train_loss": 0.5 - epoch * 0.1,
            "val_accuracy": 0.7 + epoch * 0.05,
            "learning_rate": 0.001
        })

    progress_logger.on_train_end()

    # 测试CSVLogger
    csv_logger = CSVLogger(filename="./test_logs/training.csv")
    csv_logger.on_train_begin()

    for epoch in range(2):
        csv_logger.on_epoch_end(epoch, {
            "train_loss": 0.5,
            "val_accuracy": 0.7,
            "learning_rate": 0.001
        })

    print("CSV文件创建成功")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")

    print("\n回调函数模块测试通过!")


if __name__ == "__main__":
    test_callbacks()