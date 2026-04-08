"""
日志工具模块
提供统一的日志记录功能
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logger(name: str = "pointcloud_classification",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不写入文件
        console_output: 是否输出到控制台
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 获取日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 避免日志传播到根记录器
    logger.propagate = False

    return logger


class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为JSON格式

        Args:
            record: 日志记录

        Returns:
            str: JSON格式的日志字符串
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, ensure_ascii=False)


def setup_json_logger(name: str = "pointcloud_classification",
                      log_level: str = "INFO",
                      log_file: Optional[str] = None) -> logging.Logger:
    """
    设置JSON格式的日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_file: 日志文件路径

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(f"{name}_json")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 清除现有的处理器
    logger.handlers.clear()

    # 创建JSON格式化器
    json_formatter = JsonFormatter()

    # 控制台处理器（输出JSON）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果需要）
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


class ProgressLogger:
    """进度日志记录器"""

    def __init__(self, total: int, desc: str = "进度", unit: str = "it",
                 logger: Optional[logging.Logger] = None):
        """
        初始化进度日志记录器

        Args:
            total: 总步骤数
            desc: 描述
            unit: 单位
            logger: 日志记录器，如果为None则创建新的
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = datetime.now()

        self.logger = logger or setup_logger("progress")

    def update(self, n: int = 1, status: Optional[str] = None):
        """
        更新进度

        Args:
            n: 完成的步骤数
            status: 状态信息
        """
        self.current += n
        progress = self.current / self.total * 100

        # 计算预计剩余时间
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            time_per_unit = elapsed_time / self.current
            remaining_time = time_per_unit * (self.total - self.current)
        else:
            remaining_time = 0

        # 格式化时间
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(remaining_time)

        # 记录进度
        message = f"{self.desc}: {progress:.1f}% ({self.current}/{self.total} {self.unit})"
        if status:
            message += f" - {status}"
        message += f" [已用: {elapsed_str}, 剩余: {remaining_str}]"

        self.logger.info(message)

    def finish(self, message: str = "完成"):
        """完成进度记录"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        time_str = self._format_time(total_time)

        self.logger.info(f"{self.desc} {message} - 总时间: {time_str}")

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        elif minutes > 0:
            return f"{minutes:02d}:{secs:02d}"
        else:
            return f"{secs:02d}s"


def log_experiment_config(config: Dict[str, Any], logger: logging.Logger):
    """
    记录实验配置

    Args:
        config: 配置字典
        logger: 日志记录器
    """
    logger.info("实验配置:")
    logger.info("=" * 60)

    for section, section_config in config.items():
        logger.info(f"[{section}]")

        if isinstance(section_config, dict):
            for key, value in section_config.items():
                if isinstance(value, dict) or isinstance(value, list):
                    logger.info(f"  {key}: {type(value).__name__} ({len(value)} 项)")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {section_config}")

        logger.info("")

    logger.info("=" * 60)


def log_metrics(metrics: Dict[str, Any], epoch: int, logger: logging.Logger,
                prefix: str = ""):
    """
    记录指标

    Args:
        metrics: 指标字典
        epoch: 当前epoch
        logger: 日志记录器
        prefix: 前缀
    """
    prefix_str = f"{prefix} " if prefix else ""

    logger.info(f"{prefix_str}Epoch {epoch} 指标:")
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")


def test_logger():
    """测试日志工具"""
    print("测试日志工具...")

    # 测试普通日志记录器
    logger = setup_logger(
        name="test_logger",
        log_level="DEBUG",
        log_file="./test_logs/test.log",
        console_output=True
    )

    logger.debug("这是一条调试消息")
    logger.info("这是一条信息消息")
    logger.warning("这是一条警告消息")
    logger.error("这是一条错误消息")

    # 测试JSON日志记录器
    json_logger = setup_json_logger(
        name="test_json_logger",
        log_level="INFO",
        log_file="./test_logs/test_json.log"
    )

    json_logger.info("这是一条JSON格式的日志消息", extra={"user": "test_user", "action": "test"})

    # 测试进度日志记录器
    progress = ProgressLogger(total=100, desc="测试进度", unit="步骤")
    for i in range(0, 100, 10):
        progress.update(10, f"步骤 {i+10}")
    progress.finish()

    # 测试实验配置记录
    test_config = {
        "dataset": {
            "name": "ScanObjectNN",
            "num_points": 1024,
            "batch_size": 32
        },
        "model": {
            "type": "Point Transformer",
            "num_classes": 15
        }
    }

    log_experiment_config(test_config, logger)

    # 测试指标记录
    test_metrics = {
        "accuracy": 0.95,
        "loss": 0.15,
        "f1_score": 0.94
    }

    log_metrics(test_metrics, epoch=10, logger=logger, prefix="训练")

    print("日志工具测试完成!")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")


if __name__ == "__main__":
    test_logger()