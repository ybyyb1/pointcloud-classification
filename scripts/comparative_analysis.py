#!/usr/bin/env python3
"""
对比分析框架
比较不同模型在不同数据集上的性能
"""
import os
import sys
import json
import argparse
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_environment():
    """设置环境"""
    # 确保必要的目录存在
    os.makedirs("./experiments", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)

def create_experiment_config(name: str, description: str = "") -> Dict[str, Any]:
    """
    创建实验配置

    Args:
        name: 实验名称
        description: 实验描述

    Returns:
        Dict[str, Any]: 实验配置
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{name}_{timestamp}"

    config = {
        "experiment_id": experiment_id,
        "name": name,
        "description": description,
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "status": "pending",
        "configurations": []
    }

    return config

def add_model_config(config: Dict[str, Any], model_name: str,
                    model_params: Dict[str, Any], training_params: Dict[str, Any]) -> None:
    """
    添加模型配置到实验

    Args:
        config: 实验配置
        model_name: 模型名称
        model_params: 模型参数
        training_params: 训练参数
    """
    model_config = {
        "model": model_name,
        "model_params": model_params,
        "training_params": training_params,
        "status": "pending",
        "results": None
    }

    config["configurations"].append(model_config)

def run_single_experiment(config: Dict[str, Any], model_config_idx: int,
                         data_dir: str, use_virtual_data: bool = False) -> Dict[str, Any]:
    """
    运行单个实验配置

    Args:
        config: 实验配置
        model_config_idx: 模型配置索引
        data_dir: 数据目录
        use_virtual_data: 是否使用虚拟数据

    Returns:
        Dict[str, Any]: 实验结果
    """
    import torch
    from models.model_factory import create_model
    from training.trainer import Trainer
    from config import TrainingConfig

    model_config = config["configurations"][model_config_idx]
    print(f"运行实验: {config['name']} - {model_config['model']}")

    # 设置环境变量（如果需要虚拟数据）
    if use_virtual_data:
        os.environ['SCANOBJECTNN_ALLOW_DUMMY'] = 'true'
        print("使用虚拟数据集进行测试")

    try:
        # 创建数据集
        from config import DatasetConfig, DatasetType
        from data.datasets.scanobjectnn_dataset import create_scanobjectnn_dataloader

        dataset_config = DatasetConfig(
            dataset_type=DatasetType.SCANOBJECTNN,
            data_dir=data_dir,
            num_points=1024,
            batch_size=model_config["training_params"].get("batch_size", 4)
        )

        # 创建数据加载器
        train_loader = create_scanobjectnn_dataloader(dataset_config, split="train", shuffle=True)
        val_loader = create_scanobjectnn_dataloader(dataset_config, split="test", shuffle=False)

        # 创建模型
        model_params = model_config["model_params"].copy()
        model_params["num_classes"] = len(train_loader.dataset.class_names)
        model_params["num_points"] = dataset_config.num_points

        model = create_model({
            "model_name": model_config["model"],
            **model_params
        })

        print(f"  模型: {model.model_name}")
        print(f"  参数数量: {model.count_parameters()[0]:,}")

        # 创建训练器
        train_config = TrainingConfig(
            epochs=model_config["training_params"].get("epochs", 5),
            batch_size=model_config["training_params"].get("batch_size", 4),
            learning_rate=model_config["training_params"].get("learning_rate", 0.001),
            save_checkpoint_interval=1
        )

        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # 训练
        print(f"  训练 {train_config.epochs} 个epoch...")
        history = trainer.train()

        # 评估
        results = trainer.evaluate(val_loader)

        # 收集结果
        experiment_results = {
            "model_name": model_config["model"],
            "parameters": model.count_parameters(),
            "training_history": {
                "train_loss": [float(h.get("train_loss", 0)) for h in history],
                "train_accuracy": [float(h.get("train_accuracy", 0)) for h in history],
                "val_loss": [float(h.get("val_loss", 0)) for h in history],
                "val_accuracy": [float(h.get("val_accuracy", 0)) for h in history]
            },
            "final_results": {
                "accuracy": float(results["accuracy"]),
                "loss": float(results["loss"]),
                "predictions": results.get("predictions", []).tolist() if hasattr(results.get("predictions", []), 'tolist') else results.get("predictions", []),
                "labels": results.get("labels", []).tolist() if hasattr(results.get("labels", []), 'tolist') else results.get("labels", [])
            },
            "training_time": sum(h.get("epoch_time", 0) for h in history),
            "status": "completed"
        }

        print(f"  最终准确率: {results['accuracy']:.4f}")
        print(f"  训练时间: {experiment_results['training_time']:.1f}s")

        return experiment_results

    except Exception as e:
        print(f"  实验失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            "model_name": model_config["model"],
            "status": "failed",
            "error": str(e)
        }

def run_comparative_experiment(config: Dict[str, Any], output_dir: str = "./experiments",
                             use_virtual_data: bool = False) -> Dict[str, Any]:
    """
    运行对比实验

    Args:
        config: 实验配置
        output_dir: 输出目录
        use_virtual_data: 是否使用虚拟数据

    Returns:
        Dict[str, Any]: 完整实验结果
    """
    print("=" * 80)
    print(f"运行对比实验: {config['name']}")
    print(f"实验ID: {config['experiment_id']}")
    print(f"配置数量: {len(config['configurations'])}")
    print("=" * 80)

    # 创建实验目录
    experiment_dir = os.path.join(output_dir, config["experiment_id"])
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建临时数据目录
    temp_data_dir = os.path.join(experiment_dir, "data")
    os.makedirs(temp_data_dir, exist_ok=True)

    # 运行所有配置
    all_results = []
    for i, model_config in enumerate(config["configurations"]):
        print(f"\n[{i+1}/{len(config['configurations'])}] " +
              f"配置: {model_config['model']}")

        # 运行实验
        result = run_single_experiment(
            config=config,
            model_config_idx=i,
            data_dir=temp_data_dir,
            use_virtual_data=use_virtual_data
        )

        # 更新配置状态
        config["configurations"][i]["status"] = result["status"]
        config["configurations"][i]["results"] = result

        all_results.append(result)

    # 更新实验状态
    config["status"] = "completed"
    config["completed_at"] = datetime.now().isoformat()
    config["summary"] = {
        "total_configs": len(config["configurations"]),
        "completed": sum(1 for c in config["configurations"] if c["status"] == "completed"),
        "failed": sum(1 for c in config["configurations"] if c["status"] == "failed")
    }

    # 保存实验配置和结果
    config_file = os.path.join(experiment_dir, "experiment_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n实验配置保存到: {config_file}")

    # 生成对比报告
    report = generate_comparison_report(config, experiment_dir)
    print(f"对比报告保存到: {report}")

    return config

def generate_comparison_report(config: Dict[str, Any], experiment_dir: str) -> str:
    """
    生成对比报告

    Args:
        config: 实验配置
        experiment_dir: 实验目录

    Returns:
        str: 报告文件路径
    """
    print("\n生成对比报告...")

    # 收集成功的结果
    successful_results = []
    for model_config in config["configurations"]:
        if model_config["status"] == "completed" and model_config["results"]:
            results = model_config["results"]
            successful_results.append({
                "model": model_config["model"],
                "parameters": results.get("parameters", (0, 0))[0],
                "trainable_parameters": results.get("parameters", (0, 0))[1],
                "final_accuracy": results.get("final_results", {}).get("accuracy", 0),
                "final_loss": results.get("final_results", {}).get("loss", 0),
                "training_time": results.get("training_time", 0),
                "epochs": len(results.get("training_history", {}).get("train_loss", []))
            })

    if not successful_results:
        print("没有成功的结果可以比较")
        return ""

    # 创建DataFrame
    df = pd.DataFrame(successful_results)

    # 基本统计
    report_content = f"""对比分析报告
{'=' * 80}
实验名称: {config['name']}
实验ID: {config['experiment_id']}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总配置数: {config['summary']['total_configs']}
成功数: {config['summary']['completed']}
失败数: {config['summary']['failed']}
{'=' * 80}

模型性能对比:
{df.to_string(index=False)}

性能指标:
{'=' * 80}
"""

    # 添加性能排名
    report_content += "\n准确率排名:\n"
    accuracy_rank = df.sort_values("final_accuracy", ascending=False)
    for i, (_, row) in enumerate(accuracy_rank.iterrows(), 1):
        report_content += f"  {i}. {row['model']}: {row['final_accuracy']:.4f}\n"

    report_content += "\n训练时间排名（越短越好）:\n"
    time_rank = df.sort_values("training_time")
    for i, (_, row) in enumerate(time_rank.iterrows(), 1):
        report_content += f"  {i}. {row['model']}: {row['training_time']:.1f}s\n"

    report_content += "\n参数量排名（越少越好）:\n"
    param_rank = df.sort_values("parameters")
    for i, (_, row) in enumerate(param_rank.iterrows(), 1):
        report_content += f"  {i}. {row['model']}: {row['parameters']:,} 参数\n"

    # 计算效率指标（准确率/参数比）
    df["accuracy_per_param"] = df["final_accuracy"] / df["parameters"] * 1e6
    df["accuracy_per_second"] = df["final_accuracy"] / df["training_time"]

    report_content += f"""
效率指标:
{'=' * 80}
准确率/参数比 (每百万参数):
"""
    efficiency_rank = df.sort_values("accuracy_per_param", ascending=False)
    for i, (_, row) in enumerate(efficiency_rank.iterrows(), 1):
        report_content += f"  {i}. {row['model']}: {row['accuracy_per_param']:.6f}\n"

    report_content += "\n准确率/时间比 (每秒):\n"
    time_efficiency_rank = df.sort_values("accuracy_per_second", ascending=False)
    for i, (_, row) in enumerate(time_efficiency_rank.iterrows(), 1):
        report_content += f"  {i}. {row['model']}: {row['accuracy_per_second']:.6f}\n"

    # 建议
    best_accuracy = df.loc[df["final_accuracy"].idxmax()]
    best_efficiency = df.loc[df["accuracy_per_param"].idxmax()]
    fastest = df.loc[df["training_time"].idxmin()]

    report_content += f"""
建议:
{'=' * 80}
1. 最佳准确率模型: {best_accuracy['model']} ({best_accuracy['final_accuracy']:.4f})
   - 参数量: {best_accuracy['parameters']:,}
   - 训练时间: {best_accuracy['training_time']:.1f}s

2. 最高效率模型: {best_efficiency['model']}
   - 准确率/参数比: {best_efficiency['accuracy_per_param']:.6f}
   - 适合资源受限环境

3. 最快训练模型: {fastest['model']}
   - 训练时间: {fastest['training_time']:.1f}s
   - 适合快速原型开发

4. 综合推荐:
   - 追求最高精度: {best_accuracy['model']}
   - 平衡精度和效率: {best_efficiency['model']}
   - 快速实验: {fastest['model']}
"""

    # 保存报告
    report_file = os.path.join(experiment_dir, "comparison_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # 保存CSV文件
    csv_file = os.path.join(experiment_dir, "results.csv")
    df.to_csv(csv_file, index=False)

    # 生成图表
    try:
        generate_comparison_charts(df, experiment_dir)
    except Exception as e:
        print(f"生成图表失败: {e}")

    return report_file

def generate_comparison_charts(df: pd.DataFrame, output_dir: str) -> None:
    """
    生成对比图表

    Args:
        df: 结果DataFrame
        output_dir: 输出目录
    """
    import matplotlib.pyplot as plt

    # 设置中文字体（如果需要）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # 1. 准确率对比柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["model"], df["final_accuracy"])
    plt.title("模型准确率对比")
    plt.xlabel("模型")
    plt.ylabel("准确率")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    accuracy_chart = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(accuracy_chart, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 训练时间对比
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["model"], df["training_time"])
    plt.title("模型训练时间对比")
    plt.xlabel("模型")
    plt.ylabel("训练时间 (秒)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    time_chart = os.path.join(output_dir, "training_time_comparison.png")
    plt.savefig(time_chart, dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 准确率-参数量散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df["parameters"], df["final_accuracy"], s=100, alpha=0.7)

    # 添加标签
    for i, row in df.iterrows():
        plt.annotate(row["model"],
                    (row["parameters"], row["final_accuracy"]),
                    xytext=(5, 5), textcoords='offset points')

    plt.title("准确率 vs 参数量")
    plt.xlabel("参数量")
    plt.ylabel("准确率")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_chart = os.path.join(output_dir, "accuracy_vs_parameters.png")
    plt.savefig(scatter_chart, dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 综合对比雷达图（可选）
    try:
        # 标准化数据
        from sklearn.preprocessing import MinMaxScaler

        metrics = ["final_accuracy", "training_time", "parameters"]
        scaled_data = []

        for metric in metrics:
            if metric == "training_time" or metric == "parameters":
                # 对于时间和参数量，越小越好，所以取倒数
                values = df[metric].values
                # 避免除零
                values = np.maximum(values, 1e-10)
                scaled = 1.0 / values
            else:
                scaled = df[metric].values

            # 归一化到0-1
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(scaled.reshape(-1, 1)).flatten()
            scaled_data.append(scaled)

        scaled_data = np.array(scaled_data)

        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 闭合

        for i, row in df.iterrows():
            values = scaled_data[:, i]
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=row["model"])
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(angles[:-1] * 180/np.pi, ["准确率", "训练时间", "参数量"])
        ax.set_title("模型综合对比雷达图", y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()

        radar_chart = os.path.join(output_dir, "radar_comparison.png")
        plt.savefig(radar_chart, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"生成雷达图失败: {e}")

    print(f"图表保存到: {output_dir}")

def create_predefined_experiments() -> Dict[str, Dict[str, Any]]:
    """
    创建预定义实验配置

    Returns:
        Dict[str, Dict[str, Any]]: 预定义实验配置字典
    """
    experiments = {}

    # 实验1: 基础模型对比
    exp1_config = create_experiment_config(
        name="基础模型对比",
        description="对比Point Transformer, PointNet, DGCNN在ScanObjectNN上的性能"
    )

    # Point Transformer配置
    add_model_config(exp1_config, "point_transformer",
                    {"dim": 512, "depth": 6, "num_heads": 8},
                    {"epochs": 10, "batch_size": 8, "learning_rate": 0.001})

    # PointNet配置
    add_model_config(exp1_config, "pointnet",
                    {},
                    {"epochs": 10, "batch_size": 8, "learning_rate": 0.001})

    # DGCNN配置
    add_model_config(exp1_config, "dgcnn",
                    {},
                    {"epochs": 10, "batch_size": 8, "learning_rate": 0.001})

    experiments["basic_comparison"] = exp1_config

    # 实验2: Point Transformer变体对比
    exp2_config = create_experiment_config(
        name="Point Transformer变体对比",
        description="对比不同配置的Point Transformer性能"
    )

    # 不同深度的Point Transformer
    for depth in [4, 6, 8]:
        add_model_config(exp2_config, "point_transformer",
                        {"dim": 512, "depth": depth, "num_heads": 8},
                        {"epochs": 8, "batch_size": 8, "learning_rate": 0.001})

    # 不同维度的Point Transformer
    for dim in [256, 512, 768]:
        add_model_config(exp2_config, "point_transformer",
                        {"dim": dim, "depth": 6, "num_heads": 8},
                        {"epochs": 8, "batch_size": 8, "learning_rate": 0.001})

    experiments["transformer_variants"] = exp2_config

    # 实验3: 训练策略对比
    exp3_config = create_experiment_config(
        name="训练策略对比",
        description="对比不同学习率和批次大小的效果"
    )

    model_name = "point_transformer"
    model_params = {"dim": 512, "depth": 6, "num_heads": 8}

    # 不同学习率
    for lr in [0.0001, 0.001, 0.01]:
        add_model_config(exp3_config, model_name, model_params,
                        {"epochs": 8, "batch_size": 8, "learning_rate": lr})

    # 不同批次大小
    for bs in [4, 8, 16]:
        add_model_config(exp3_config, model_name, model_params,
                        {"epochs": 8, "batch_size": bs, "learning_rate": 0.001})

    experiments["training_strategies"] = exp3_config

    return experiments

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="对比分析框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行预定义的基础模型对比实验（使用虚拟数据）
  python comparative_analysis.py --experiment basic_comparison --virtual

  # 运行自定义实验
  python comparative_analysis.py --custom --models point_transformer,pointnet,dgcnn

  # 列出所有预定义实验
  python comparative_analysis.py --list

  # 生成实验报告（不运行实验）
  python comparative_analysis.py --report-only --experiment-dir ./experiments/exp_001
        """
    )

    parser.add_argument("--experiment", type=str,
                       choices=["basic_comparison", "transformer_variants", "training_strategies"],
                       help="运行预定义实验")
    parser.add_argument("--custom", action="store_true", help="运行自定义实验")
    parser.add_argument("--models", type=str, help="自定义模型列表，逗号分隔")
    parser.add_argument("--virtual", action="store_true",
                       help="使用虚拟数据集（用于测试）")
    parser.add_argument("--list", action="store_true", help="列出预定义实验")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不运行实验")
    parser.add_argument("--experiment-dir", type=str, help="实验目录（用于报告生成）")
    parser.add_argument("--output-dir", type=str, default="./experiments",
                       help="输出目录")

    args = parser.parse_args()

    # 初始化环境
    setup_environment()

    if args.list:
        # 列出预定义实验
        experiments = create_predefined_experiments()
        print("预定义实验:")
        print("=" * 80)
        for exp_id, exp_config in experiments.items():
            print(f"{exp_id}:")
            print(f"  名称: {exp_config['name']}")
            print(f"  描述: {exp_config['description']}")
            print(f"  配置数: {len(exp_config['configurations'])}")
            print()
        return

    if args.report_only:
        # 只生成报告
        if not args.experiment_dir:
            print("错误: 需要指定实验目录")
            sys.exit(1)

        if not os.path.exists(args.experiment_dir):
            print(f"错误: 实验目录不存在: {args.experiment_dir}")
            sys.exit(1)

        # 加载实验配置
        config_file = os.path.join(args.experiment_dir, "experiment_config.json")
        if not os.path.exists(config_file):
            print(f"错误: 实验配置文件不存在: {config_file}")
            sys.exit(1)

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 生成报告
        report = generate_comparison_report(config, args.experiment_dir)
        print(f"报告生成完成: {report}")
        return

    # 确定要运行的实验
    if args.experiment:
        # 运行预定义实验
        experiments = create_predefined_experiments()
        if args.experiment not in experiments:
            print(f"错误: 未知的实验: {args.experiment}")
            sys.exit(1)

        config = experiments[args.experiment]

    elif args.custom:
        # 运行自定义实验
        if not args.models:
            print("错误: 自定义实验需要指定模型列表")
            sys.exit(1)

        models = args.models.split(',')
        config = create_experiment_config(
            name="自定义实验",
            description=f"对比模型: {', '.join(models)}"
        )

        # 添加每个模型的配置
        for model_name in models:
            add_model_config(config, model_name, {},
                           {"epochs": 5, "batch_size": 8, "learning_rate": 0.001})

    else:
        # 默认运行基础模型对比
        experiments = create_predefined_experiments()
        config = experiments["basic_comparison"]

    # 运行实验
    results = run_comparative_experiment(
        config=config,
        output_dir=args.output_dir,
        use_virtual_data=args.virtual
    )

    # 总结
    print("\n" + "=" * 80)
    print("实验完成!")
    print(f"实验ID: {results['experiment_id']}")
    print(f"成功配置: {results['summary']['completed']}/{results['summary']['total_configs']}")
    print(f"实验目录: {os.path.join(args.output_dir, results['experiment_id'])}")
    print("=" * 80)

if __name__ == "__main__":
    main()