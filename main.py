
#!/usr/bin/env python3
"""
点云分类系统主入口
基于点云数据的室内场景三维物体分类系统
"""

import os
import sys
import argparse
import json
import yaml
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig, load_config, save_config
from data.datasets import create_scanobjectnn_dataloader, create_s3dis_classification_dataset
from models.model_factory import ModelFactory, create_model
from training.trainer import Trainer, KaggleTrainer
from training.metrics import AccuracyMetric, ConfusionMatrixMetric
from training.callbacks import EarlyStopping, ModelCheckpoint, ProgressLogger
from utils.logger import setup_logger


def download_scanobjectnn(args):
    """下载ScanObjectNN数据集"""
    from data.datasets.scanobjectnn_dataset import ScanObjectNNDataset
    from config import DatasetConfig, DatasetType

    print("下载ScanObjectNN数据集...")

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.SCANOBJECTNN,
        data_dir=args.data_dir if args.data_dir else "./data/scanobjectnn",
        scanobjectnn_version=args.version if args.version else "main_split",
        num_points=args.num_points if args.num_points else 1024,
        batch_size=args.batch_size if args.batch_size else 32
    )

    # 创建数据集（会自动下载）
    dataset = ScanObjectNNDataset(config, split="train")
    print(f"ScanObjectNN数据集下载完成")
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(dataset.class_names)}")

    # 保存统计信息
    stats = dataset.get_statistics()
    stats_file = os.path.join(config.data_dir, "dataset_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"数据集统计信息保存到: {stats_file}")


def build_s3dis_dataset(args):
    """从S3DIS构建分类数据集"""
    print("从S3DIS构建分类数据集...")

    from config import DatasetConfig, DatasetType

    # 创建配置
    config = DatasetConfig(
        dataset_type=DatasetType.S3DIS,
        data_dir=args.data_dir if args.data_dir else "./data/s3dis_classification",
        s3dis_area=args.areas if args.areas else [1, 2, 3, 4, 5, 6],
        s3dis_classes_to_include=args.classes if args.classes else [
            "table", "chair", "sofa", "bookcase", "board"
        ],
        num_points=args.num_points if args.num_points else 1024,
        batch_size=args.batch_size if args.batch_size else 32,
        train_ratio=args.train_ratio if args.train_ratio else 0.7,
        val_ratio=args.val_ratio if args.val_ratio else 0.15,
        test_ratio=args.test_ratio if args.test_ratio else 0.15
    )

    # 创建S3DIS分类数据集
    from data.datasets.s3dis_dataset import create_s3dis_classification_dataset as create_s3dis
    create_s3dis(config, force_reprocess=args.force)

    print("S3DIS分类数据集构建完成")


def train_model(args):
    """训练模型"""
    print(f"训练模型: {args.model}")

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = SystemConfig()

        # 更新配置
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.model_dir:
            config.model_dir = args.model_dir
        if args.output_dir:
            config.output_dir = args.output_dir

        config.training.epochs = args.epochs if args.epochs else config.training.epochs
        config.training.learning_rate = args.learning_rate if args.learning_rate else config.training.learning_rate
        config.training.batch_size = args.batch_size if args.batch_size else config.training.batch_size

    # 设置数据集类型
    if args.dataset == "scanobjectnn":
        from config import DatasetType
        config.dataset.dataset_type = DatasetType.SCANOBJECTNN
    elif args.dataset == "s3dis":
        from config import DatasetType
        config.dataset.dataset_type = DatasetType.S3DIS
    elif args.dataset == "custom":
        from config import DatasetType
        config.dataset.dataset_type = DatasetType.CUSTOM

    # 创建数据加载器
    if config.dataset.dataset_type.value == "scanobjectnn":
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader
        train_loader = create_scanobjectnn_dataloader(config.dataset, split="train")
        val_loader = create_scanobjectnn_dataloader(config.dataset, split="test")  # ScanObjectNN使用test作为验证集
    elif config.dataset.dataset_type.value == "s3dis":
        from data.datasets.s3dis_dataset import S3DISDataset
        train_dataset = S3DISDataset(config.dataset, split="train")
        val_dataset = S3DISDataset(config.dataset, split="val")

        import torch
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers
        )
    else:
        raise ValueError(f"不支持的数据集类型: {config.dataset.dataset_type}")

    # 创建模型
    model_config = {
        "model_name": args.model,
        "num_classes": len(train_loader.dataset.class_names),
        "num_points": config.dataset.num_points
    }

    model = create_model(model_config)
    print(f"创建模型: {model.model_name}")
    print(f"参数数量: {model.count_parameters()[0]:,}")

    # 创建训练器
    if args.kaggle:
        trainer = KaggleTrainer(
            model=model,
            config=config.training,
            train_loader=train_loader,
            val_loader=val_loader
        )
    else:
        trainer = Trainer(
            model=model,
            config=config.training,
            train_loader=train_loader,
            val_loader=val_loader
        )

    # 添加回调函数
    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=args.early_stopping_patience if args.early_stopping_patience else 20,
            mode="max"
        )
        trainer.add_callback(early_stopping)

    # 添加模型检查点
    checkpoint_dir = os.path.join(config.training.checkpoint_dir, args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.pth"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
    trainer.add_callback(model_checkpoint)

    # 添加进度日志
    progress_logger = ProgressLogger(
        log_file=os.path.join(checkpoint_dir, "training.log"),
        verbose=1
    )
    trainer.add_callback(progress_logger)

    # 开始训练
    history = trainer.train()

    # 保存训练历史
    history_file = os.path.join(checkpoint_dir, "training_history.json")
    trainer.save_history(history_file)

    # 绘制训练历史图表
    plot_file = os.path.join(checkpoint_dir, "training_history.png")
    trainer.plot_history(plot_file)

    print(f"训练完成! 最佳验证准确率: {trainer.best_metric:.4f}")
    print(f"检查点保存在: {checkpoint_dir}")


def evaluate_model(args):
    """评估模型"""
    print(f"评估模型: {args.checkpoint}")

    # 加载模型
    from models.model_factory import load_model
    model = load_model(args.checkpoint)

    # 加载配置（从检查点或单独配置文件）
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint.get("model_config", {})

    # 创建数据加载器
    if args.dataset == "scanobjectnn":
        from config import DatasetConfig, DatasetType
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader

        config = DatasetConfig(
            dataset_type=DatasetType.SCANOBJECTNN,
            data_dir=args.data_dir if args.data_dir else "./data/scanobjectnn",
            num_points=model_config.get("num_points", 1024),
            batch_size=args.batch_size if args.batch_size else 32
        )

        test_loader = create_scanobjectnn_dataloader(config, split="test", shuffle=False)

    elif args.dataset == "s3dis":
        from config import DatasetConfig, DatasetType
        from data.datasets.s3dis_dataset import S3DISDataset

        config = DatasetConfig(
            dataset_type=DatasetType.S3DIS,
            data_dir=args.data_dir if args.data_dir else "./data/s3dis_classification",
            num_points=model_config.get("num_points", 1024),
            batch_size=args.batch_size if args.batch_size else 32
        )

        test_dataset = S3DISDataset(config, split="test")
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    # 创建训练器用于评估
    from config import TrainingConfig
    eval_config = TrainingConfig(
        batch_size=args.batch_size if args.batch_size else 32
    )

    from training.trainer import Trainer
    trainer = Trainer(
        model=model,
        config=eval_config,
        train_loader=test_loader,  # 只用于评估，train_loader不会使用
        val_loader=test_loader
    )

    # 评估
    results = trainer.evaluate(test_loader)

    # 计算混淆矩阵
    confusion_matrix = ConfusionMatrixMetric(num_classes=len(test_loader.dataset.class_names))
    confusion_matrix.update(
        torch.tensor(results["predictions"]),
        torch.tensor(results["labels"])
    )

    # 获取分类报告
    class_names = test_loader.dataset.class_names
    report = confusion_matrix.get_classification_report(class_names)

    # 保存评估结果
    eval_dir = os.path.dirname(args.checkpoint)
    if not eval_dir:
        eval_dir = "./evaluations"

    os.makedirs(eval_dir, exist_ok=True)

    # 保存结果
    results_file = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "accuracy": results["accuracy"],
            "loss": results["loss"],
            "class_names": class_names,
            "classification_report": report,
            "predictions": results["predictions"].tolist(),
            "labels": results["labels"].tolist()
        }, f, indent=2, ensure_ascii=False)

    print(f"评估结果保存到: {results_file}")

    # 绘制混淆矩阵
    cm_plot_file = os.path.join(eval_dir, "confusion_matrix.png")
    confusion_matrix.plot_confusion_matrix(class_names, cm_plot_file)

    print(f"混淆矩阵保存到: {cm_plot_file}")


def compare_models(args):
    """比较多个模型"""
    print("比较多个模型...")

    # 模型配置列表
    model_configs = {}

    if args.models:
        # 解析模型列表
        models = args.models.split(',')
        for model_name in models:
            model_configs[model_name] = {
                "num_classes": args.num_classes if args.num_classes else 15,
                "num_points": args.num_points if args.num_points else 1024
            }
    else:
        # 使用默认模型
        default_models = ["point_transformer", "pointnet", "dgcnn"]
        for model_name in default_models:
            model_configs[model_name] = {
                "num_classes": 15,
                "num_points": 1024
            }

    # 比较模型
    from models.model_factory import ModelFactory
    results = ModelFactory.compare_models(model_configs)

    # 输出比较结果
    print("\n模型比较结果:")
    print("=" * 80)
    print(f"{'模型名称':<20} {'状态':<15} {'参数数量':<15} {'输出形状':<20}")
    print("-" * 80)

    for model_name, result in results.items():
        if result["status"] == "success":
            print(f"{model_name:<20} {'成功':<15} {result['total_params']:<15,} {str(result['output_shape']):<20}")
        else:
            print(f"{model_name:<20} {'失败':<15} {'-':<15} {result['status']:<20}")

    # 保存比较结果
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式
            serializable_results = {}
            for model_name, result in results.items():
                serializable_results[model_name] = {
                    "status": result["status"],
                    "total_params": result["total_params"],
                    "trainable_params": result["trainable_params"],
                    "output_shape": str(result["output_shape"]) if result["output_shape"] is not None else None,
                    "config": result["config"]
                }

            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n比较结果保存到: {args.output}")


def visualize_data(args):
    """可视化数据"""
    print(f"可视化数据: {args.dataset}")

    if args.dataset == "scanobjectnn":
        from config import DatasetConfig, DatasetType
        from data.datasets.scanobjectnn_dataset import ScanObjectNNDataset

        config = DatasetConfig(
            dataset_type=DatasetType.SCANOBJECTNN,
            data_dir=args.data_dir if args.data_dir else "./data/scanobjectnn",
            num_points=args.num_points if args.num_points else 1024
        )

        dataset = ScanObjectNNDataset(config, split="train")

    elif args.dataset == "s3dis":
        from config import DatasetConfig, DatasetType
        from data.datasets.s3dis_dataset import S3DISDataset

        config = DatasetConfig(
            dataset_type=DatasetType.S3DIS,
            data_dir=args.data_dir if args.data_dir else "./data/s3dis_classification",
            num_points=args.num_points if args.num_points else 1024
        )

        dataset = S3DISDataset(config, split="train")

    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    # 可视化样本
    output_dir = args.output_dir if args.output_dir else "./visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 可视化指定样本或随机样本
    if args.sample_indices:
        indices = [int(idx) for idx in args.sample_indices.split(',')]
    else:
        # 随机选择几个样本
        import random
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    for idx in indices:
        if 0 <= idx < len(dataset):
            output_file = os.path.join(output_dir, f"sample_{idx}.png")
            dataset.visualize_sample(idx, save_path=output_file)
            print(f"样本 {idx} 可视化保存到: {output_file}")

    # 可视化类别分布
    stats = dataset.get_statistics()
    class_dist = stats.get("class_distribution", {})

    if class_dist:
        import matplotlib.pyplot as plt

        classes = list(class_dist.keys())
        counts = list(class_dist.values())

        plt.figure(figsize=(12, 6))
        plt.bar(classes, counts)
        plt.xlabel('类别')
        plt.ylabel('样本数量')
        plt.title(f'{args.dataset} 数据集类别分布')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        dist_file = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"类别分布图保存到: {dist_file}")


def create_config(args):
    """创建配置文件"""
    print("创建配置文件...")

    # 创建默认配置
    config = SystemConfig()

    # 根据参数更新配置
    if args.project_name:
        config.project_name = args.project_name

    if args.data_dir:
        config.data_dir = args.data_dir
        config.dataset.data_dir = args.data_dir

    if args.model:
        from config import ModelType
        try:
            config.model.model_type = ModelType(args.model)
        except ValueError:
            print(f"警告: 未知的模型类型 {args.model}，使用默认值")

    if args.dataset:
        from config import DatasetType
        try:
            config.dataset.dataset_type = DatasetType(args.dataset)
        except ValueError:
            print(f"警告: 未知的数据集类型 {args.dataset}，使用默认值")

    # 保存配置文件
    output_file = args.output if args.output else "./configs/custom_config.yaml"
    output_dir = os.path.dirname(output_file)
    if output_dir:  # 只有当目录非空时才创建
        os.makedirs(output_dir, exist_ok=True)

    save_config(config, output_file)
    print(f"配置文件保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于点云数据的室内场景三维物体分类系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载ScanObjectNN数据集
  python main.py download-scanobjectnn --data_dir ./data/scanobjectnn

  # 从S3DIS构建分类数据集
  python main.py build-s3dis --data_dir ./data/s3dis_classification

  # 训练Point Transformer模型
  python main.py train --model point_transformer --dataset scanobjectnn --epochs 100

  # 评估模型
  python main.py evaluate --checkpoint ./checkpoints/best_model.pth --dataset scanobjectnn

  # 比较多个模型
  python main.py compare --models point_transformer,pointnet,dgcnn

  # 可视化数据
  python main.py visualize --dataset scanobjectnn --num_samples 5

  # 创建配置文件
  python main.py create-config --project_name "My Point Cloud Project"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # download-scanobjectnn 命令
    download_parser = subparsers.add_parser("download-scanobjectnn", help="下载ScanObjectNN数据集")
    download_parser.add_argument("--data_dir", type=str, help="数据保存目录")
    download_parser.add_argument("--version", type=str, default="main_split", help="数据集版本")
    download_parser.add_argument("--num_points", type=int, default=1024, help="点云采样点数")
    download_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")

    # build-s3dis 命令
    build_parser = subparsers.add_parser("build-s3dis", help="从S3DIS构建分类数据集")
    build_parser.add_argument("--data_dir", type=str, help="数据保存目录")
    build_parser.add_argument("--areas", type=int, nargs="+", help="S3DIS区域列表")
    build_parser.add_argument("--classes", type=str, nargs="+", help="包含的类别")
    build_parser.add_argument("--num_points", type=int, default=1024, help="点云采样点数")
    build_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    build_parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    build_parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    build_parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    build_parser.add_argument("--force", action="store_true", help="强制重新处理")

    # train 命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model", type=str, default="point_transformer",
                            help="模型类型: point_transformer, pointnet, pointnet2, dgcnn")
    train_parser.add_argument("--dataset", type=str, default="scanobjectnn",
                            help="数据集: scanobjectnn, s3dis, custom")
    train_parser.add_argument("--config", type=str, help="配置文件路径")
    train_parser.add_argument("--data_dir", type=str, help="数据目录")
    train_parser.add_argument("--model_dir", type=str, help="模型保存目录")
    train_parser.add_argument("--output_dir", type=str, help="输出目录")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--learning_rate", type=float, help="学习率")
    train_parser.add_argument("--batch_size", type=int, help="批次大小")
    train_parser.add_argument("--early_stopping", action="store_true", help="启用早停")
    train_parser.add_argument("--early_stopping_patience", type=int, default=20, help="早停耐心值")
    train_parser.add_argument("--experiment_name", type=str, default="exp_001", help="实验名称")
    train_parser.add_argument("--kaggle", action="store_true", help="启用Kaggle优化")

    # evaluate 命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    eval_parser.add_argument("--dataset", type=str, default="scanobjectnn",
                           help="数据集: scanobjectnn, s3dis")
    eval_parser.add_argument("--data_dir", type=str, help="数据目录")
    eval_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")

    # compare 命令
    compare_parser = subparsers.add_parser("compare", help="比较多个模型")
    compare_parser.add_argument("--models", type=str, help="模型列表，逗号分隔")
    compare_parser.add_argument("--num_classes", type=int, default=15, help="类别数量")
    compare_parser.add_argument("--num_points", type=int, default=1024, help="点云点数")
    compare_parser.add_argument("--output", type=str, help="输出文件路径")

    # visualize 命令
    viz_parser = subparsers.add_parser("visualize", help="可视化数据")
    viz_parser.add_argument("--dataset", type=str, required=True,
                          help="数据集: scanobjectnn, s3dis")
    viz_parser.add_argument("--data_dir", type=str, help="数据目录")
    viz_parser.add_argument("--num_points", type=int, default=1024, help="点云采样点数")
    viz_parser.add_argument("--num_samples", type=int, default=5, help="可视化样本数量")
    viz_parser.add_argument("--sample_indices", type=str, help="指定样本索引，逗号分隔")
    viz_parser.add_argument("--output_dir", type=str, help="输出目录")

    # create-config 命令
    config_parser = subparsers.add_parser("create-config", help="创建配置文件")
    config_parser.add_argument("--project_name", type=str, help="项目名称")
    config_parser.add_argument("--data_dir", type=str, help="数据目录")
    config_parser.add_argument("--model", type=str, help="模型类型")
    config_parser.add_argument("--dataset", type=str, help="数据集类型")
    config_parser.add_argument("--output", type=str, help="输出文件路径")

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 执行命令
    try:
        if args.command == "download-scanobjectnn":
            download_scanobjectnn(args)
        elif args.command == "build-s3dis":
            build_s3dis_dataset(args)
        elif args.command == "train":
            train_model(args)
        elif args.command == "evaluate":
            evaluate_model(args)
        elif args.command == "compare":
            compare_models(args)
        elif args.command == "visualize":
            visualize_data(args)
        elif args.command == "create-config":
            create_config(args)
        else:
            print(f"未知命令: {args.command}")
            parser.print_help()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()