"""
基础模型类
定义点云分类模型的通用接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json
import os


class BaseModel(nn.Module, ABC):
    """点云分类模型的抽象基类"""

    def __init__(self, num_classes: int = 15, input_channels: int = 3):
        """
        初始化模型

        Args:
            num_classes: 类别数量
            input_channels: 输入通道数（通常为3，表示XYZ坐标）
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入点云，形状为 (B, N, C) 或 (B, C, N)

        Returns:
            torch.Tensor: 分类得分，形状为 (B, num_classes)
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """
        获取模型参数

        Returns:
            Dict[str, Any]: 参数字典
        """
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "model_name": self.model_name,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def save(self, filepath: str, save_optimizer: bool = False,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None,
             epoch: Optional[int] = None,
             metrics: Optional[Dict[str, float]] = None) -> None:
        """
        保存模型

        Args:
            filepath: 保存路径
            save_optimizer: 是否保存优化器状态
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前训练轮数
            metrics: 评估指标
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 准备保存数据
        save_data = {
            "model_state_dict": self.state_dict(),
            "model_parameters": self.get_parameters(),
            "model_name": self.model_name,
        }

        if save_optimizer and optimizer is not None:
            save_data["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            save_data["scheduler_state_dict"] = scheduler.state_dict()

        if epoch is not None:
            save_data["epoch"] = epoch

        if metrics is not None:
            save_data["metrics"] = metrics

        # 保存模型
        torch.save(save_data, filepath)
        print(f"模型保存到: {filepath}")

        # 同时保存JSON格式的配置信息
        config_file = filepath.replace(".pth", "_config.json")
        config_data = {
            "model_name": self.model_name,
            "parameters": self.get_parameters(),
            "epoch": epoch,
            "metrics": metrics,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str, load_optimizer: bool = False,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        加载模型

        Args:
            filepath: 模型文件路径
            load_optimizer: 是否加载优化器状态
            optimizer: 优化器
            scheduler: 学习率调度器

        Returns:
            Dict[str, Any]: 加载的信息
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        # 加载模型
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint["model_state_dict"])

        # 移动到当前设备
        device = next(self.parameters()).device
        self.to(device)

        # 加载优化器状态
        if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 返回加载的信息
        loaded_info = {
            "model_name": checkpoint.get("model_name", self.model_name),
            "parameters": checkpoint.get("model_parameters", {}),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

        print(f"模型从 {filepath} 加载成功")
        print(f"  模型名称: {loaded_info['model_name']}")
        print(f"  训练轮数: {loaded_info['epoch']}")
        if loaded_info['metrics']:
            print(f"  评估指标: {loaded_info['metrics']}")

        return loaded_info

    def summary(self) -> str:
        """
        生成模型摘要

        Returns:
            str: 模型摘要字符串
        """
        import io
        from contextlib import redirect_stdout

        # 捕获print输出
        f = io.StringIO()
        with redirect_stdout(f):
            print(f"模型名称: {self.model_name}")
            print(f"输入通道: {self.input_channels}")
            print(f"输出类别: {self.num_classes}")

            params = self.get_parameters()
            print(f"总参数: {params['total_parameters']:,}")
            print(f"可训练参数: {params['trainable_parameters']:,}")

            # 打印层信息
            print("\n层结构:")
            for name, module in self.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {module.__class__.__name__} ({num_params:,} 参数)")

        return f.getvalue()

    def count_parameters(self) -> Tuple[int, int]:
        """
        计算模型参数数量

        Returns:
            Tuple[int, int]: (总参数数量, 可训练参数数量)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def freeze_layers(self, layer_names: Optional[list] = None) -> None:
        """
        冻结指定层

        Args:
            layer_names: 要冻结的层名称列表，如果为None则冻结所有层
        """
        if layer_names is None:
            # 冻结所有层
            for param in self.parameters():
                param.requires_grad = False
            print("所有层已冻结")
        else:
            # 冻结指定层
            for name, param in self.named_parameters():
                for layer_name in layer_names:
                    if layer_name in name:
                        param.requires_grad = False
                        print(f"层 {name} 已冻结")
                        break

    def unfreeze_layers(self, layer_names: Optional[list] = None) -> None:
        """
        解冻指定层

        Args:
            layer_names: 要解冻的层名称列表，如果为None则解冻所有层
        """
        if layer_names is None:
            # 解冻所有层
            for param in self.parameters():
                param.requires_grad = True
            print("所有层已解冻")
        else:
            # 解冻指定层
            for name, param in self.named_parameters():
                for layer_name in layer_names:
                    if layer_name in name:
                        param.requires_grad = True
                        print(f"层 {name} 已解冻")
                        break

    def get_layer_outputs(self, x: torch.Tensor, layer_names: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        获取指定层的输出

        Args:
            x: 输入张量
            layer_names: 要获取输出的层名称列表，如果为None则获取所有层

        Returns:
            Dict[str, torch.Tensor]: 层名称到输出的映射
        """
        outputs = {}

        # 定义钩子函数
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output.detach()
            return hook

        # 注册钩子
        hooks = []
        for name, module in self.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            self(x)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return outputs


def test_base_model():
    """测试基础模型类"""
    print("测试基础模型类...")

    # 创建一个简单的测试模型
    class TestModel(BaseModel):
        def __init__(self, num_classes=10):
            super().__init__(num_classes=num_classes)
            self.fc = nn.Linear(100, num_classes)

        def forward(self, x):
            # 假设输入已经是合适的形状
            return self.fc(x)

    # 创建模型实例
    model = TestModel(num_classes=15)

    # 测试参数获取
    params = model.get_parameters()
    print(f"模型参数: {params}")

    # 测试参数计数
    total_params, trainable_params = model.count_parameters()
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 测试模型摘要
    summary = model.summary()
    print(f"模型摘要:\n{summary}")

    # 测试前向传播
    test_input = torch.randn(4, 100)  # 批量大小4，特征维度100
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 测试保存和加载
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pth")

        # 保存模型
        model.save(model_path, epoch=10, metrics={"accuracy": 0.95})
        print(f"模型保存到: {model_path}")

        # 创建新模型并加载
        new_model = TestModel(num_classes=15)
        loaded_info = new_model.load(model_path)
        print(f"加载信息: {loaded_info}")

        # 验证加载的模型
        new_output = new_model(test_input)
        print(f"原始输出与加载后输出是否一致: {torch.allclose(output, new_output, rtol=1e-4)}")

    print("基础模型类测试通过!")


if __name__ == "__main__":
    test_base_model()