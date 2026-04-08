"""
结果分析器模块
用于分析和比较不同模型的实验结果
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings


class ResultAnalyzer:
    """结果分析器"""

    def __init__(self, results_dir: str = "./experiments"):
        """
        初始化结果分析器

        Args:
            results_dir: 结果目录
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def load_experiment_results(self, experiment_path: str) -> Dict[str, Any]:
        """
        加载实验结果

        Args:
            experiment_path: 实验目录路径

        Returns:
            Dict[str, Any]: 实验结果
        """
        results = {}

        # 加载配置文件
        config_path = os.path.join(experiment_path, "config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                results["config"] = yaml.safe_load(f)

        # 加载训练历史
        history_path = os.path.join(experiment_path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                results["history"] = json.load(f)

        # 加载评估结果
        eval_path = os.path.join(experiment_path, "evaluation_results.json")
        if os.path.exists(eval_path):
            with open(eval_path, 'r', encoding='utf-8') as f:
                results["evaluation"] = json.load(f)

        # 加载CSV日志
        csv_path = os.path.join(experiment_path, "training.csv")
        if os.path.exists(csv_path):
            results["csv_data"] = pd.read_csv(csv_path)

        # 实验元数据
        results["experiment_name"] = os.path.basename(experiment_path)
        results["experiment_path"] = experiment_path

        return results

    def find_all_experiments(self, base_dir: Optional[str] = None) -> List[str]:
        """
        查找所有实验目录

        Args:
            base_dir: 基础目录，如果为None则使用self.results_dir

        Returns:
            List[str]: 实验目录列表
        """
        if base_dir is None:
            base_dir = self.results_dir

        experiments = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # 检查是否包含配置文件
                config_path = os.path.join(item_path, "config.yaml")
                if os.path.exists(config_path):
                    experiments.append(item_path)

        return experiments

    def compare_experiments(self, experiment_paths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        比较多个实验

        Args:
            experiment_paths: 实验路径列表，如果为None则查找所有实验

        Returns:
            pd.DataFrame: 比较结果
        """
        if experiment_paths is None:
            experiment_paths = self.find_all_experiments()

        comparison_data = []

        for exp_path in experiment_paths:
            try:
                results = self.load_experiment_results(exp_path)

                # 提取关键指标
                exp_data = {
                    "experiment": os.path.basename(exp_path),
                    "path": exp_path,
                }

                # 配置信息
                config = results.get("config", {})
                if config:
                    # 模型配置
                    model_config = config.get("model", {})
                    if model_config:
                        exp_data["model_type"] = model_config.get("model_type", "unknown")
                        exp_data["num_points"] = model_config.get("num_points", 0)

                    # 训练配置
                    training_config = config.get("training", {})
                    if training_config:
                        exp_data["learning_rate"] = training_config.get("learning_rate", 0.0)
                        exp_data["batch_size"] = training_config.get("batch_size", 0)
                        exp_data["epochs"] = training_config.get("epochs", 0)
                        exp_data["optimizer"] = training_config.get("optimizer", "unknown")

                # 训练历史
                history = results.get("history", {})
                if history:
                    exp_data["final_train_loss"] = history.get("train_loss", [0])[-1] if history.get("train_loss") else 0.0
                    exp_data["final_val_loss"] = history.get("val_loss", [0])[-1] if history.get("val_loss") else 0.0
                    exp_data["final_train_acc"] = history.get("train_accuracy", [0])[-1] if history.get("train_accuracy") else 0.0
                    exp_data["final_val_acc"] = history.get("val_accuracy", [0])[-1] if history.get("val_accuracy") else 0.0

                    # 最佳指标
                    if history.get("val_accuracy"):
                        exp_data["best_val_acc"] = max(history["val_accuracy"])
                        exp_data["best_epoch"] = history["val_accuracy"].index(exp_data["best_val_acc"]) + 1

                # 评估结果
                evaluation = results.get("evaluation", {})
                if evaluation:
                    exp_data["test_accuracy"] = evaluation.get("accuracy", 0.0)
                    exp_data["test_loss"] = evaluation.get("loss", 0.0)
                    exp_data["best_val_acc"] = evaluation.get("best_validation_accuracy", exp_data.get("best_val_acc", 0.0))
                    exp_data["best_epoch"] = evaluation.get("best_epoch", exp_data.get("best_epoch", 0))

                comparison_data.append(exp_data)

            except Exception as e:
                warnings.warn(f"加载实验 {exp_path} 失败: {e}")

        # 创建DataFrame
        df = pd.DataFrame(comparison_data)

        # 按测试准确率排序
        if not df.empty and "test_accuracy" in df.columns:
            df = df.sort_values("test_accuracy", ascending=False)

        return df

    def analyze_training_dynamics(self, experiment_path: str) -> Dict[str, Any]:
        """
        分析训练动态

        Args:
            experiment_path: 实验目录路径

        Returns:
            Dict[str, Any]: 训练动态分析结果
        """
        results = self.load_experiment_results(experiment_path)
        history = results.get("history", {})

        analysis = {
            "experiment": os.path.basename(experiment_path),
            "convergence_analysis": {},
            "overfitting_analysis": {},
            "learning_rate_analysis": {},
        }

        # 收敛性分析
        if history.get("train_loss") and history.get("val_loss"):
            train_loss = history["train_loss"]
            val_loss = history["val_loss"]

            analysis["convergence_analysis"] = {
                "final_train_loss": train_loss[-1],
                "final_val_loss": val_loss[-1],
                "loss_decrease_ratio": (train_loss[0] - train_loss[-1]) / train_loss[0] if train_loss[0] > 0 else 0,
                "convergence_epoch": self._find_convergence_epoch(val_loss),
                "is_converged": self._check_convergence(val_loss),
            }

        # 过拟合分析
        if history.get("train_accuracy") and history.get("val_accuracy"):
            train_acc = history["train_accuracy"]
            val_acc = history["val_accuracy"]

            gap = max(train_acc) - max(val_acc) if val_acc else 0
            analysis["overfitting_analysis"] = {
                "max_train_acc": max(train_acc) if train_acc else 0,
                "max_val_acc": max(val_acc) if val_acc else 0,
                "accuracy_gap": gap,
                "is_overfitting": gap > 0.1,  # 准确率差距超过10%认为过拟合
                "overfitting_score": gap,
            }

        # 学习率分析
        if history.get("learning_rate"):
            lr_history = history["learning_rate"]
            analysis["learning_rate_analysis"] = {
                "initial_lr": lr_history[0] if lr_history else 0,
                "final_lr": lr_history[-1] if lr_history else 0,
                "lr_decay_ratio": (lr_history[0] - lr_history[-1]) / lr_history[0] if lr_history[0] > 0 else 0,
                "lr_schedule_type": self._analyze_lr_schedule(lr_history),
            }

        return analysis

    def _find_convergence_epoch(self, loss_history: List[float], threshold: float = 0.001) -> int:
        """
        找到收敛的epoch

        Args:
            loss_history: 损失历史
            threshold: 收敛阈值

        Returns:
            int: 收敛的epoch
        """
        if len(loss_history) < 2:
            return 0

        for i in range(1, len(loss_history)):
            if abs(loss_history[i] - loss_history[i-1]) < threshold:
                return i

        return len(loss_history) - 1

    def _check_convergence(self, loss_history: List[float], window: int = 5, threshold: float = 0.001) -> bool:
        """
        检查是否收敛

        Args:
            loss_history: 损失历史
            window: 滑动窗口大小
            threshold: 收敛阈值

        Returns:
            bool: 是否收敛
        """
        if len(loss_history) < window:
            return False

        recent_losses = loss_history[-window:]
        variance = np.var(recent_losses)

        return variance < threshold

    def _analyze_lr_schedule(self, lr_history: List[float]) -> str:
        """
        分析学习率调度策略

        Args:
            lr_history: 学习率历史

        Returns:
            str: 调度策略类型
        """
        if len(lr_history) < 2:
            return "constant"

        lr_changes = []
        for i in range(1, len(lr_history)):
            if lr_history[i] != lr_history[i-1]:
                lr_changes.append((i, lr_history[i] / lr_history[i-1]))

        if not lr_changes:
            return "constant"

        # 分析变化模式
        change_ratios = [ratio for _, ratio in lr_changes]

        if all(ratio < 1 for ratio in change_ratios):
            return "decay"
        elif any(ratio > 1 for ratio in change_ratios):
            return "cyclic"
        else:
            return "step_decay"

    def generate_comparison_report(self, comparison_df: pd.DataFrame,
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成比较报告

        Args:
            comparison_df: 比较DataFrame
            output_path: 输出文件路径

        Returns:
            Dict[str, Any]: 比较报告
        """
        if comparison_df.empty:
            return {"error": "没有可比较的数据"}

        report = {
            "summary": {
                "total_experiments": len(comparison_df),
                "best_model": None,
                "best_accuracy": 0.0,
                "average_accuracy": 0.0,
            },
            "model_performance": {},
            "detailed_comparison": comparison_df.to_dict(orient="records"),
        }

        # 计算统计信息
        if "test_accuracy" in comparison_df.columns:
            report["summary"]["best_accuracy"] = comparison_df["test_accuracy"].max()
            report["summary"]["average_accuracy"] = comparison_df["test_accuracy"].mean()
            report["summary"]["std_accuracy"] = comparison_df["test_accuracy"].std()

            # 找到最佳模型
            best_idx = comparison_df["test_accuracy"].idxmax()
            best_model = comparison_df.loc[best_idx]
            report["summary"]["best_model"] = best_model["experiment"]

        # 按模型类型分组统计
        if "model_type" in comparison_df.columns:
            model_groups = comparison_df.groupby("model_type")
            for model_type, group in model_groups:
                report["model_performance"][model_type] = {
                    "count": len(group),
                    "avg_accuracy": group["test_accuracy"].mean() if "test_accuracy" in group.columns else 0.0,
                    "best_accuracy": group["test_accuracy"].max() if "test_accuracy" in group.columns else 0.0,
                    "avg_training_time": None,  # 可以添加训练时间信息
                }

        # 保存报告
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def plot_comparison(self, comparison_df: pd.DataFrame,
                       output_path: Optional[str] = None) -> None:
        """
        绘制比较图表

        Args:
            comparison_df: 比较DataFrame
            output_path: 输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if comparison_df.empty:
                print("没有数据可绘制")
                return

            # 设置样式
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 模型准确率比较
            if "model_type" in comparison_df.columns and "test_accuracy" in comparison_df.columns:
                ax = axes[0, 0]
                sns.boxplot(data=comparison_df, x="model_type", y="test_accuracy", ax=ax)
                ax.set_xlabel("模型类型")
                ax.set_ylabel("测试准确率")
                ax.set_title("不同模型的测试准确率比较")
                ax.tick_params(axis='x', rotation=45)

            # 2. 学习率与准确率关系
            if "learning_rate" in comparison_df.columns and "test_accuracy" in comparison_df.columns:
                ax = axes[0, 1]
                sns.scatterplot(data=comparison_df, x="learning_rate", y="test_accuracy",
                               hue="model_type" if "model_type" in comparison_df.columns else None,
                               size="batch_size" if "batch_size" in comparison_df.columns else None,
                               ax=ax)
                ax.set_xlabel("学习率")
                ax.set_ylabel("测试准确率")
                ax.set_title("学习率与准确率关系")
                ax.set_xscale('log')

            # 3. 批次大小与准确率关系
            if "batch_size" in comparison_df.columns and "test_accuracy" in comparison_df.columns:
                ax = axes[1, 0]
                sns.lineplot(data=comparison_df, x="batch_size", y="test_accuracy",
                            hue="model_type" if "model_type" in comparison_df.columns else None,
                            marker='o', ax=ax)
                ax.set_xlabel("批次大小")
                ax.set_ylabel("测试准确率")
                ax.set_title("批次大小与准确率关系")

            # 4. 训练epoch与准确率关系
            if "epochs" in comparison_df.columns and "test_accuracy" in comparison_df.columns:
                ax = axes[1, 1]
                sns.regplot(data=comparison_df, x="epochs", y="test_accuracy", ax=ax)
                ax.set_xlabel("训练轮数")
                ax.set_ylabel("测试准确率")
                ax.set_title("训练轮数与准确率关系")

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"比较图表保存到: {output_path}")
            else:
                plt.show()

        except ImportError:
            print("警告: matplotlib或seaborn未安装，无法绘制图表")
        except Exception as e:
            print(f"绘制图表失败: {e}")


def analyze_experiment_results(experiment_dir: str = "./experiments",
                              output_dir: Optional[str] = None):
    """
    分析实验结果的主函数

    Args:
        experiment_dir: 实验目录
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, "analysis")

    os.makedirs(output_dir, exist_ok=True)

    analyzer = ResultAnalyzer(experiment_dir)

    print("分析实验结果...")

    # 查找所有实验
    experiments = analyzer.find_all_experiments()
    print(f"找到 {len(experiments)} 个实验")

    if not experiments:
        print("没有找到实验")
        return

    # 比较实验
    comparison_df = analyzer.compare_experiments(experiments)

    # 保存比较结果
    comparison_csv = os.path.join(output_dir, "experiment_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
    print(f"比较结果保存到: {comparison_csv}")

    # 生成报告
    report_path = os.path.join(output_dir, "comparison_report.json")
    report = analyzer.generate_comparison_report(comparison_df, report_path)
    print(f"比较报告保存到: {report_path}")

    # 绘制比较图表
    plot_path = os.path.join(output_dir, "comparison_plots.png")
    analyzer.plot_comparison(comparison_df, plot_path)

    # 分析每个实验的训练动态
    for exp_path in experiments[:5]:  # 限制分析前5个实验
        try:
            exp_name = os.path.basename(exp_path)
            dynamics = analyzer.analyze_training_dynamics(exp_path)

            dynamics_path = os.path.join(output_dir, f"{exp_name}_dynamics.json")
            with open(dynamics_path, 'w', encoding='utf-8') as f:
                json.dump(dynamics, f, indent=2, ensure_ascii=False)

            print(f"实验 {exp_name} 的训练动态分析保存到: {dynamics_path}")

        except Exception as e:
            print(f"分析实验 {exp_path} 失败: {e}")

    print(f"\n分析完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="分析实验结果")
    parser.add_argument("--experiment_dir", type=str, default="./experiments",
                       help="实验目录")
    parser.add_argument("--output_dir", type=str, help="输出目录")

    args = parser.parse_args()

    analyze_experiment_results(args.experiment_dir, args.output_dir)