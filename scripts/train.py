#!/usr/bin/env python3
"""
训练脚本
用于训练点云分类模型
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SystemConfig, load_config, save_config
from data.datasets import create_scanobjectnn_dataloader
from data.datasets.s3dis_dataset import S3DISDataset
from models.model_factory import create_model
from training.trainer import Trainer, KaggleTrainer
from training.callbacks import EarlyStopping, ModelCheckpoint, ProgressLogger, CSVLogger
from training.metrics import ConfusionMatrixMetric
from utils.logger import setup_logger, log_experiment_config


def setup_experiment(config: SystemConfig, experiment_name: str):
    """设置实验环境"""
    # 创建实验目录
    experiment_dir = os.path.join(config.training.checkpoint_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建日志记录器
    log_file = os.path.join(experiment_dir, "training.log")
    logger = setup_logger(
        name=f"train_{experiment_name}",
        log_level=config.ui.log_level,
        log_file=log_file,
        console_output=config.ui.console_log
    )

    # 保存配置文件
    config_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_path)

    # 记录实验配置
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"实验目录: {experiment_dir}")
    log_experiment_config(config.__dict__, logger)

    return experiment_dir, logger


def create_data_loaders(config, logger):
    """创建数据加载器"""
    logger.info("创建数据加载器...")

    dataset_type = config.dataset.dataset_type.value

    if dataset_type == "scanobjectnn":
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader

        logger.info(f"使用ScanObjectNN数据集")
        train_loader = create_scanobjectnn_dataloader(
            config.dataset, split="train", shuffle=True
        )
        val_loader = create_scanobjectnn_dataloader(
            config.dataset, split="test", shuffle=False  # ScanObjectNN使用test作为验证集
        )
        test_loader = create_scanobjectnn_dataloader(
            config.dataset, split="test", shuffle=False
        )

    elif dataset_type == "s3dis":
        logger.info(f"使用S3DIS数据集")

        train_dataset = S3DISDataset(config.dataset, split="train")
        val_dataset = S3DISDataset(config.dataset, split="val")
        test_dataset = S3DISDataset(config.dataset, split="test")

        import torch
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            pin_memory=True
        )

    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    logger.info(f"训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")
    logger.info(f"测试集: {len(test_loader.dataset)} 样本")

    # 记录类别信息
    class_names = train_loader.dataset.class_names
    logger.info(f"类别数量: {len(class_names)}")
    logger.info(f"类别名称: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def create_model_with_config(config, num_classes, logger):
    """根据配置创建模型"""
    logger.info("创建模型...")

    model_type = config.model.model_type.value
    model_config = {
        "model_name": model_type,
        "num_classes": num_classes,
        "num_points": config.dataset.num_points,
    }

    # 添加模型特定配置
    if model_type == "point_transformer":
        model_config.update({
            "dim": config.model.point_transformer_dim,
            "depth": config.model.point_transformer_depth,
            "num_heads": config.model.point_transformer_heads,
            "mlp_ratio": config.model.point_transformer_mlp_ratio,
            "drop_rate": config.model.point_transformer_drop_rate,
        })
    elif model_type == "pointnet":
        model_config.update({
            "mlp_layers": config.model.pointnet_mlp_layers,
            "use_tnet": config.model.pointnet_use_tnet,
            "dropout": config.model.pointnet_dropout,
        })
    elif model_type == "dgcnn":
        model_config.update({
            "k": config.model.dgcnn_k,
            "emb_dims": config.model.dgcnn_emb_dims,
            "dropout": config.model.dgcnn_dropout,
        })

    model = create_model(model_config)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"模型: {model.model_name}")
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    return model


def setup_callbacks(config, experiment_dir, logger):
    """设置回调函数"""
    logger.info("设置回调函数...")

    callbacks = []

    # 早停回调
    if config.training.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=config.training.early_stopping_patience,
            mode="max",
            min_delta=0.001
        )
        callbacks.append(early_stopping)
        logger.info(f"早停回调: 耐心值={config.training.early_stopping_patience}")

    # 模型检查点回调
    checkpoint_path = os.path.join(experiment_dir, "best_model.pth")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False
    )
    callbacks.append(model_checkpoint)
    logger.info(f"模型检查点: {checkpoint_path}")

    # 进度日志回调
    progress_logger = ProgressLogger(
        log_file=os.path.join(experiment_dir, "progress.log"),
        verbose=1
    )
    callbacks.append(progress_logger)

    # CSV日志回调
    csv_logger = CSVLogger(
        filename=os.path.join(experiment_dir, "training.csv"),
        separator=",",
        append=False
    )
    callbacks.append(csv_logger)

    return callbacks


def train_model(config_path: str = None,
                experiment_name: str = "exp_001",
                use_kaggle: bool = False,
                num_epochs: int = None):
    """
    训练模型主函数

    Args:
        config_path: 配置文件路径
        experiment_name: 实验名称
        use_kaggle: 是否使用Kaggle优化
        num_epochs: 训练轮数
    """
    # 加载配置
    if config_path:
        config = load_config(config_path)
    else:
        config = SystemConfig()

    # 更新训练轮数
    if num_epochs:
        config.training.epochs = num_epochs

    # 设置实验环境
    experiment_dir, logger = setup_experiment(config, experiment_name)

    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader, class_names = create_data_loaders(config, logger)

        # 创建模型
        model = create_model_with_config(config, len(class_names), logger)

        # 创建训练器
        if use_kaggle:
            logger.info("使用Kaggle训练器")
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

        # 设置回调函数
        callbacks = setup_callbacks(config, experiment_dir, logger)
        for callback in callbacks:
            trainer.add_callback(callback)

        # 开始训练
        logger.info("开始训练...")
        history = trainer.train()

        # 保存训练历史
        history_path = os.path.join(experiment_dir, "training_history.json")
        trainer.save_history(history_path)

        # 绘制训练历史图表
        plot_path = os.path.join(experiment_dir, "training_history.png")
        trainer.plot_history(plot_path)

        # 在测试集上评估最佳模型
        logger.info("在测试集上评估模型...")
        test_results = trainer.evaluate(test_loader)

        # 计算混淆矩阵
        confusion_matrix = ConfusionMatrixMetric(num_classes=len(class_names))
        import torch
        confusion_matrix.update(
            torch.tensor(test_results["predictions"]),
            torch.tensor(test_results["labels"])
        )

        # 获取分类报告
        report = confusion_matrix.get_classification_report(class_names)

        # 保存评估结果
        eval_results = {
            "accuracy": test_results["accuracy"],
            "loss": test_results["loss"],
            "class_names": class_names,
            "classification_report": report,
            "best_validation_accuracy": trainer.best_metric,
            "best_epoch": trainer.best_epoch + 1,
        }

        eval_path = os.path.join(experiment_dir, "evaluation_results.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        # 绘制混淆矩阵
        cm_path = os.path.join(experiment_dir, "confusion_matrix.png")
        confusion_matrix.plot_confusion_matrix(class_names, cm_path)

        logger.info("训练完成!")
        logger.info(f"最佳验证准确率: {trainer.best_metric:.4f}")
        logger.info(f"测试准确率: {test_results['accuracy']:.4f}")
        logger.info(f"结果保存到: {experiment_dir}")

        return {
            "experiment_dir": experiment_dir,
            "best_accuracy": trainer.best_metric,
            "test_accuracy": test_results["accuracy"],
            "history": history,
            "evaluation": eval_results
        }

    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="训练点云分类模型")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--experiment", type=str, default="exp_001", help="实验名称")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--data_dir", type=str, help="数据目录")
    parser.add_argument("--model", type=str, help="模型类型")
    parser.add_argument("--dataset", type=str, help="数据集类型")
    parser.add_argument("--kaggle", action="store_true", help="使用Kaggle优化")
    parser.add_argument("--gpu", type=int, default=0, help="GPU设备ID")

    args = parser.parse_args()

    # 设置GPU
    if args.gpu >= 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print(f"使用GPU: {args.gpu} - {torch.cuda.get_device_name(args.gpu)}")
        else:
            print("警告: GPU不可用，将使用CPU")

    # 训练模型
    try:
        results = train_model(
            config_path=args.config,
            experiment_name=args.experiment,
            use_kaggle=args.kaggle,
            num_epochs=args.epochs
        )

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"实验目录: {results['experiment_dir']}")
        print(f"最佳验证准确率: {results['best_accuracy']:.4f}")
        print(f"测试准确率: {results['test_accuracy']:.4f}")
        print("=" * 60)

    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()