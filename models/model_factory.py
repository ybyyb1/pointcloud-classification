"""
模型工厂
用于创建和管理各种点云分类模型
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from .base_model import BaseModel
from .point_transformer import PointTransformer, SimplifiedPointTransformer, create_point_transformer
from .pointnet import PointNet, create_pointnet
from .pointnet2 import PointNet2, create_pointnet2
from .dgcnn import DGCNN, create_dgcnn


class ModelFactory:
    """模型工厂类"""

    # 模型注册表
    _model_registry = {
        "point_transformer": PointTransformer,
        "point_transformer_simple": SimplifiedPointTransformer,
        "pointnet": PointNet,
        "pointnet2": PointNet2,
        "dgcnn": DGCNN,
    }

    # 模型创建函数注册表
    _creator_registry = {
        "point_transformer": create_point_transformer,
        "pointnet": create_pointnet,
        "pointnet2": create_pointnet2,
        "dgcnn": create_dgcnn,
    }

    @classmethod
    def register_model(cls, name: str, model_class: type, creator_func: Optional[callable] = None):
        """
        注册新模型

        Args:
            name: 模型名称
            model_class: 模型类
            creator_func: 模型创建函数（可选）
        """
        cls._model_registry[name] = model_class
        if creator_func is not None:
            cls._creator_registry[name] = creator_func

    @classmethod
    def list_models(cls) -> list:
        """
        列出所有可用的模型

        Returns:
            list: 模型名称列表
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_model_class(cls, name: str) -> type:
        """
        获取模型类

        Args:
            name: 模型名称

        Returns:
            type: 模型类
        """
        name = name.lower()
        if name not in cls._model_registry:
            raise ValueError(f"未知的模型: {name}。可用模型: {list(cls._model_registry.keys())}")
        return cls._model_registry[name]

    @classmethod
    def create_model(cls, config: Union[Dict[str, Any], str]) -> BaseModel:
        """
        创建模型

        Args:
            config: 配置字典或模型名称

        Returns:
            BaseModel: 创建的模型
        """
        if isinstance(config, str):
            # 如果config是字符串，使用默认配置
            model_name = config
            config = {"model_name": model_name}

        # 获取模型名称
        model_name = config.get("model_name", "point_transformer").lower()

        # 检查是否有专门的创建函数
        if model_name in cls._creator_registry:
            creator_func = cls._creator_registry[model_name]
            # 获取创建函数的参数签名
            import inspect
            sig = inspect.signature(creator_func)
            param_names = list(sig.parameters.keys())

            # 从配置中移除model_name
            filtered_config = {k: v for k, v in config.items() if k != "model_name"}

            # 如果创建函数只有一个参数且名为'config'，直接传递字典
            if len(param_names) == 1 and param_names[0] == 'config':
                return creator_func(filtered_config)
            else:
                # 否则进行参数过滤
                filtered_kwargs = {}
                for key, value in filtered_config.items():
                    if key in param_names:
                        filtered_kwargs[key] = value
                    else:
                        # 警告但忽略未知参数
                        print(f"警告: 模型 {model_name} 的创建函数忽略未知参数 '{key}'")

                return creator_func(**filtered_kwargs)

        # 使用模型类直接创建
        if model_name not in cls._model_registry:
            raise ValueError(f"未知的模型: {model_name}")

        model_class = cls._model_registry[model_name]

        # 使用inspect获取模型构造函数的有效参数
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = list(sig.parameters.keys())

        # 过滤配置参数，只保留模型构造函数接受的参数
        model_params = {}
        for key, value in config.items():
            if key in valid_params and key != 'self':
                model_params[key] = value
            elif key not in ['model_name', 'self']:
                # 警告但忽略未知参数
                print(f"警告: 模型 {model_name} 忽略未知参数 '{key}'")

        # 创建模型实例
        return model_class(**model_params)

    @classmethod
    def create_model_from_checkpoint(cls, checkpoint_path: str) -> BaseModel:
        """
        从检查点创建模型

        Args:
            checkpoint_path: 检查点文件路径

        Returns:
            BaseModel: 加载的模型
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 获取模型信息
        model_name = checkpoint.get("model_name", "point_transformer")
        model_params = checkpoint.get("model_parameters", {})

        # 创建模型
        model = cls.create_model({"model_name": model_name, **model_params})

        # 加载权重
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    @classmethod
    def compare_models(cls, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        比较多个模型

        Args:
            configs: 模型配置字典，键为模型名称，值为配置

        Returns:
            Dict[str, Dict[str, Any]]: 比较结果
        """
        results = {}

        for model_name, config in configs.items():
            try:
                # 创建模型
                model = cls.create_model({**config, "model_name": model_name})

                # 计算参数
                total_params, trainable_params = model.count_parameters()

                # 测试前向传播
                test_input = torch.randn(2, 1024, 3)  # 小型测试输入
                with torch.no_grad():
                    output = model(test_input)

                # 存储结果
                results[model_name] = {
                    "model": model,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "output_shape": output.shape,
                    "config": config,
                    "status": "success"
                }

                print(f"模型 {model_name}: {total_params:,} 参数，输出形状 {output.shape}")

            except Exception as e:
                results[model_name] = {
                    "model": None,
                    "total_params": 0,
                    "trainable_params": 0,
                    "output_shape": None,
                    "config": config,
                    "status": f"failed: {e}"
                }

                print(f"模型 {model_name} 创建失败: {e}")

        return results


def create_model(config: Union[Dict[str, Any], str]) -> BaseModel:
    """
    创建模型的快捷函数

    Args:
        config: 配置字典或模型名称

    Returns:
        BaseModel: 创建的模型
    """
    return ModelFactory.create_model(config)


def load_model(checkpoint_path: str) -> BaseModel:
    """
    从检查点加载模型的快捷函数

    Args:
        checkpoint_path: 检查点文件路径

    Returns:
        BaseModel: 加载的模型
    """
    return ModelFactory.create_model_from_checkpoint(checkpoint_path)


def test_model_factory():
    """测试模型工厂"""
    print("测试模型工厂...")

    # 列出可用模型
    models = ModelFactory.list_models()
    print(f"可用模型: {models}")

    # 测试创建各种模型
    test_configs = {
        "point_transformer": {
            "num_classes": 15,
            "num_points": 1024,
            "dim": 512,
            "depth": 6,
        },
        "point_transformer_simple": {
            "num_classes": 15,
            "num_points": 1024,
            "dim": 256,
            "depth": 4,
        },
    }

    print("\n创建和比较模型:")

    # 比较模型
    results = ModelFactory.compare_models(test_configs)

    print("\n模型详细信息:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  状态: {result['status']}")
        if result['status'] == "success":
            print(f"  总参数: {result['total_params']:,}")
            print(f"  可训练参数: {result['trainable_params']:,}")
            print(f"  输出形状: {result['output_shape']}")

    # 测试模型创建函数
    print("\n测试模型创建函数:")
    try:
        model = create_model("point_transformer")
        print(f"创建模型成功: {model.__class__.__name__}")

        # 测试检查点保存和加载
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_model.pth")
            model.save(checkpoint_path, epoch=0, metrics={})

            loaded_model = load_model(checkpoint_path)
            print(f"加载模型成功: {loaded_model.__class__.__name__}")

    except Exception as e:
        print(f"模型创建测试失败: {e}")

    print("\n模型工厂测试完成!")


if __name__ == "__main__":
    test_model_factory()