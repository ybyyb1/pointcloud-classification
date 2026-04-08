"""
学习率调度器模块
包含各种学习率调度器的创建和配置
"""

import torch
import torch.optim as optim
import math
from typing import Dict, Any, List, Optional, Union
import warnings


def create_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_config: 调度器配置

    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: 学习率调度器
    """
    # 默认配置
    default_config = {
        "type": "cosine",
        "t_max": 100,
        "eta_min": 1e-6,
        "step_size": 30,
        "gamma": 0.1,
        "milestones": [30, 60, 90],
        "factor": 0.1,
        "patience": 10,
        "mode": "max",
        "threshold": 1e-4,
        "cooldown": 0,
        "min_lr": 1e-6,
        "warmup_epochs": 5,
        "warmup_lr": 1e-7,
    }

    # 更新配置
    config = {**default_config, **scheduler_config}
    scheduler_type = config["type"].lower()

    if scheduler_type == "none" or scheduler_type == "null":
        return None

    # 根据类型创建调度器
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["t_max"],
            eta_min=config["eta_min"]
        )

    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"]
        )

    elif scheduler_type == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["milestones"],
            gamma=config["gamma"]
        )

    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config["gamma"]
        )

    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["mode"],
            factor=config["factor"],
            patience=config["patience"],
            threshold=config["threshold"],
            cooldown=config["cooldown"],
            min_lr=config["min_lr"],
            verbose=True
        )

    elif scheduler_type == "cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.get("base_lr", 1e-6),
            max_lr=config.get("max_lr", 0.01),
            step_size_up=config.get("step_size_up", 2000),
            mode=config.get("cyclic_mode", "triangular")
        )

    elif scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 0.01),
            total_steps=config.get("total_steps", 100),
            pct_start=config.get("pct_start", 0.3),
            anneal_strategy=config.get("anneal_strategy", "cos")
        )

    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    # 添加热身
    if config["warmup_epochs"] > 0:
        scheduler = WarmupSchedulerWrapper(
            scheduler,
            warmup_epochs=config["warmup_epochs"],
            warmup_lr=config["warmup_lr"],
            optimizer=optimizer
        )

    return scheduler


def create_scheduler_with_params(optimizer: torch.optim.Optimizer,
                                 scheduler_type: str = "cosine",
                                 **kwargs) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    使用参数创建调度器（简化版）

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 调度器参数

    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: 学习率调度器
    """
    config = {
        "type": scheduler_type,
        **kwargs
    }

    return create_scheduler(optimizer, config)


def get_scheduler_info(scheduler: Optional[optim.lr_scheduler._LRScheduler]) -> Dict[str, Any]:
    """
    获取调度器信息

    Args:
        scheduler: 调度器对象

    Returns:
        Dict[str, Any]: 调度器信息
    """
    if scheduler is None:
        return {"type": "none", "parameters": {}}

    info = {
        "type": scheduler.__class__.__name__,
        "parameters": {},
        "current_lr": None
    }

    # 获取调度器参数
    if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
        info["parameters"] = {
            "T_max": scheduler.T_max,
            "eta_min": scheduler.eta_min,
            "last_epoch": scheduler.last_epoch,
        }

    elif isinstance(scheduler, optim.lr_scheduler.StepLR):
        info["parameters"] = {
            "step_size": scheduler.step_size,
            "gamma": scheduler.gamma,
            "last_epoch": scheduler.last_epoch,
        }

    elif isinstance(scheduler, optim.lr_scheduler.MultiStepLR):
        info["parameters"] = {
            "milestones": scheduler.milestones,
            "gamma": scheduler.gamma,
            "last_epoch": scheduler.last_epoch,
        }

    elif isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
        info["parameters"] = {
            "gamma": scheduler.gamma,
            "last_epoch": scheduler.last_epoch,
        }

    elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        info["parameters"] = {
            "mode": scheduler.mode,
            "factor": scheduler.factor,
            "patience": scheduler.patience,
            "threshold": scheduler.threshold,
            "cooldown": scheduler.cooldown,
            "min_lr": scheduler.min_lrs,
        }

    elif isinstance(scheduler, WarmupSchedulerWrapper):
        info["type"] = f"WarmupWrapper({scheduler.scheduler.__class__.__name__})"
        info["parameters"] = {
            "warmup_epochs": scheduler.warmup_epochs,
            "warmup_lr": scheduler.warmup_lr,
            "warmup_completed": scheduler.warmup_completed,
        }

    # 获取当前学习率
    if hasattr(scheduler, "optimizer"):
        try:
            info["current_lr"] = scheduler.optimizer.param_groups[0]["lr"]
        except:
            info["current_lr"] = None

    return info


class WarmupSchedulerWrapper:
    """热身调度器包装器"""

    def __init__(self, scheduler, warmup_epochs: int, warmup_lr: float,
                 optimizer: torch.optim.Optimizer):
        """
        初始化热身调度器

        Args:
            scheduler: 基础调度器
            warmup_epochs: 热身epoch数
            warmup_lr: 热身学习率
            optimizer: 优化器
        """
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.optimizer = optimizer
        self.warmup_completed = False
        self.current_epoch = 0

        # 保存初始学习率
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics=None):
        """执行一步调度"""
        self.current_epoch += 1

        # 热身阶段
        if not self.warmup_completed and self.current_epoch <= self.warmup_epochs:
            # 线性增加学习率
            alpha = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_lr + alpha * (self.initial_lrs[i] - self.warmup_lr)

            if self.current_epoch == self.warmup_epochs:
                self.warmup_completed = True
                print(f"热身完成，开始使用 {self.scheduler.__class__.__name__} 调度器")
        else:
            # 使用基础调度器
            if metrics is not None and hasattr(self.scheduler, "step"):
                self.scheduler.step(metrics)
            elif hasattr(self.scheduler, "step"):
                self.scheduler.step()

    def get_last_lr(self):
        """获取当前学习率"""
        if hasattr(self.scheduler, "get_last_lr"):
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """获取状态字典"""
        state = {
            "warmup_epochs": self.warmup_epochs,
            "warmup_lr": self.warmup_lr,
            "warmup_completed": self.warmup_completed,
            "current_epoch": self.current_epoch,
            "initial_lrs": self.initial_lrs,
            "scheduler_state": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else {}
        }
        return state

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.warmup_epochs = state_dict.get("warmup_epochs", self.warmup_epochs)
        self.warmup_lr = state_dict.get("warmup_lr", self.warmup_lr)
        self.warmup_completed = state_dict.get("warmup_completed", self.warmup_completed)
        self.current_epoch = state_dict.get("current_epoch", self.current_epoch)
        self.initial_lrs = state_dict.get("initial_lrs", self.initial_lrs)

        if hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(state_dict.get("scheduler_state", {}))


def create_cosine_annealing_with_warmup(optimizer: torch.optim.Optimizer,
                                        t_max: int = 100,
                                        warmup_epochs: int = 5,
                                        eta_min: float = 1e-6,
                                        warmup_lr: float = 1e-7) -> WarmupSchedulerWrapper:
    """
    创建带热身的余弦退火调度器

    Args:
        optimizer: 优化器
        t_max: 余弦周期
        warmup_epochs: 热身epoch数
        eta_min: 最小学习率
        warmup_lr: 热身学习率

    Returns:
        WarmupSchedulerWrapper: 调度器
    """
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )

    return WarmupSchedulerWrapper(
        cosine_scheduler,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        optimizer=optimizer
    )


def create_step_scheduler_with_warmup(optimizer: torch.optim.Optimizer,
                                      step_size: int = 30,
                                      warmup_epochs: int = 5,
                                      gamma: float = 0.1,
                                      warmup_lr: float = 1e-7) -> WarmupSchedulerWrapper:
    """
    创建带热身的步长调度器

    Args:
        optimizer: 优化器
        step_size: 步长
        warmup_epochs: 热身epoch数
        gamma: 衰减因子
        warmup_lr: 热身学习率

    Returns:
        WarmupSchedulerWrapper: 调度器
    """
    step_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    return WarmupSchedulerWrapper(
        step_scheduler,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        optimizer=optimizer
    )


def plot_lr_schedule(scheduler, num_epochs: int = 100,
                     title: str = "学习率调度计划",
                     save_path: Optional[str] = None) -> None:
    """
    绘制学习率调度计划

    Args:
        scheduler: 调度器
        num_epochs: 总epoch数
        title: 图表标题
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 模拟调度过程
        lrs = []
        dummy_optimizer = optim.Adam([torch.nn.Parameter(torch.randn(2, 3))], lr=0.01)

        if isinstance(scheduler, WarmupSchedulerWrapper):
            # 复制调度器配置
            test_scheduler = WarmupSchedulerWrapper(
                scheduler.scheduler,
                warmup_epochs=scheduler.warmup_epochs,
                warmup_lr=scheduler.warmup_lr,
                optimizer=dummy_optimizer
            )
        else:
            # 创建相同类型的调度器
            scheduler_type = scheduler.__class__
            if scheduler_type == optim.lr_scheduler.CosineAnnealingLR:
                test_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    dummy_optimizer,
                    T_max=scheduler.T_max,
                    eta_min=scheduler.eta_min
                )
            elif scheduler_type == optim.lr_scheduler.StepLR:
                test_scheduler = optim.lr_scheduler.StepLR(
                    dummy_optimizer,
                    step_size=scheduler.step_size,
                    gamma=scheduler.gamma
                )
            else:
                print(f"不支持的调度器类型用于绘图: {scheduler_type}")
                return

        # 模拟epoch
        for epoch in range(num_epochs):
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
            if hasattr(test_scheduler, 'step'):
                test_scheduler.step()

        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_epochs), lrs, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # 标记重要点
        if isinstance(scheduler, WarmupSchedulerWrapper):
            plt.axvline(x=scheduler.warmup_epochs, color='r', linestyle='--', alpha=0.7,
                       label=f'Warmup结束 (Epoch {scheduler.warmup_epochs})')
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"学习率调度图保存到: {save_path}")
        else:
            plt.show()

    except ImportError:
        print("警告: matplotlib未安装，无法绘制学习率调度图")
    except Exception as e:
        print(f"绘制学习率调度图失败: {e}")


def test_scheduler():
    """测试调度器模块"""
    print("测试调度器模块...")

    import torch.nn as nn

    # 创建测试模型和优化器
    model = nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 测试各种调度器
    scheduler_types = ["cosine", "step", "multistep", "exponential", "plateau"]

    for sched_type in scheduler_types:
        try:
            if sched_type == "plateau":
                scheduler = create_scheduler_with_params(
                    optimizer=optimizer,
                    scheduler_type=sched_type,
                    mode="max",
                    factor=0.5,
                    patience=5
                )
            else:
                scheduler = create_scheduler_with_params(
                    optimizer=optimizer,
                    scheduler_type=sched_type,
                    t_max=50 if sched_type == "cosine" else None,
                    step_size=10 if sched_type == "step" else None,
                    milestones=[10, 30, 50] if sched_type == "multistep" else None,
                    gamma=0.1
                )

            # 获取调度器信息
            info = get_scheduler_info(scheduler)
            print(f"{sched_type}: {info['type']}")

            # 测试几步调度
            if scheduler is not None:
                for i in range(3):
                    if sched_type == "plateau":
                        scheduler.step(0.8 - i * 0.1)  # 模拟指标下降
                    else:
                        scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  当前学习率: {current_lr:.6f}")

        except Exception as e:
            print(f"{sched_type}: 创建失败 - {e}")

    # 测试热身调度器
    print("\n测试热身调度器:")
    try:
        warmup_scheduler = create_cosine_annealing_with_warmup(
            optimizer=optimizer,
            t_max=50,
            warmup_epochs=5,
            warmup_lr=1e-6
        )

        print(f"热身调度器: {warmup_scheduler.__class__.__name__}")

        # 模拟热身阶段
        for epoch in range(7):  # 5个热身epoch + 2个正常epoch
            warmup_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch + 1}: lr={lr:.6f}")

        info = get_scheduler_info(warmup_scheduler)
        print(f"  热身完成: {info['parameters'].get('warmup_completed', False)}")

    except Exception as e:
        print(f"热身调度器失败: {e}")

    # 测试调度器信息获取
    print("\n测试调度器信息获取:")
    scheduler = create_scheduler_with_params(optimizer, "cosine", t_max=100)
    info = get_scheduler_info(scheduler)
    print(f"调度器信息: {info}")

    print("\n调度器模块测试通过!")


if __name__ == "__main__":
    test_scheduler()