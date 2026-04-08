"""
训练指标模块
包含各种训练指标的实现
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any
import warnings


class Metric:
    """指标基类"""

    def __init__(self, name: str = "metric"):
        """
        初始化指标

        Args:
            name: 指标名称
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """重置指标"""
        raise NotImplementedError

    def update(self, value: Any, n: int = 1) -> None:
        """
        更新指标

        Args:
            value: 指标值
            n: 样本数量
        """
        raise NotImplementedError

    def compute(self) -> Any:
        """
        计算指标

        Returns:
            指标值
        """
        raise NotImplementedError

    @property
    def avg(self) -> float:
        """平均指标值"""
        return self.compute()


class AverageMetric(Metric):
    """平均值指标"""

    def __init__(self, name: str = "average"):
        """
        初始化平均值指标

        Args:
            name: 指标名称
        """
        super().__init__(name)

    def reset(self) -> None:
        """重置指标"""
        self.value = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """
        更新指标

        Args:
            value: 指标值
            n: 样本数量
        """
        self.value += value * n
        self.count += n

    def compute(self) -> float:
        """
        计算平均值

        Returns:
            平均值
        """
        if self.count == 0:
            return 0.0
        return self.value / self.count


class AccuracyMetric(Metric):
    """准确率指标"""

    def __init__(self, name: str = "accuracy"):
        """
        初始化准确率指标

        Args:
            name: 指标名称
        """
        super().__init__(name)

    def reset(self) -> None:
        """重置指标"""
        self.correct = 0
        self.total = 0

    def update(self, value: float, n: int = 1) -> None:
        """
        更新准确率

        Args:
            value: 准确率值（0-1之间）
            n: 样本数量
        """
        self.correct += int(value * n)
        self.total += n

    def compute(self) -> float:
        """
        计算准确率

        Returns:
            准确率
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def update_from_predictions(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        从预测结果更新准确率

        Args:
            predictions: 预测结果，形状为 (N,) 或 (N, C)
            targets: 目标标签，形状为 (N,)
        """
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        correct = (predictions == targets).sum().item()
        total = targets.size(0)

        self.correct += correct
        self.total += total


class ConfusionMatrixMetric(Metric):
    """混淆矩阵指标"""

    def __init__(self, num_classes: int, name: str = "confusion_matrix"):
        """
        初始化混淆矩阵指标

        Args:
            num_classes: 类别数量
            name: 指标名称
        """
        super().__init__(name)
        self.num_classes = num_classes

    def reset(self) -> None:
        """重置混淆矩阵"""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        更新混淆矩阵

        Args:
            predictions: 预测结果，形状为 (N,) 或 (N, C)
            targets: 目标标签，形状为 (N,)
        """
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for i in range(len(predictions_np)):
            self.matrix[targets_np[i], predictions_np[i]] += 1

    def compute(self) -> np.ndarray:
        """
        获取混淆矩阵

        Returns:
            混淆矩阵
        """
        return self.matrix

    def get_precision(self, average: str = "macro") -> float:
        """
        计算精确率

        Args:
            average: 平均方式，"macro", "micro", 或 "weighted"

        Returns:
            精确率
        """
        return self._get_classification_metric("precision", average)

    def get_recall(self, average: str = "macro") -> float:
        """
        计算召回率

        Args:
            average: 平均方式，"macro", "micro", 或 "weighted"

        Returns:
            召回率
        """
        return self._get_classification_metric("recall", average)

    def get_f1_score(self, average: str = "macro") -> float:
        """
        计算F1分数

        Args:
            average: 平均方式，"macro", "micro", 或 "weighted"

        Returns:
            F1分数
        """
        return self._get_classification_metric("f1", average)

    def _get_classification_metric(self, metric: str, average: str) -> float:
        """
        计算分类指标

        Args:
            metric: 指标类型，"precision", "recall", 或 "f1"
            average: 平均方式

        Returns:
            指标值
        """
        # 计算每个类别的TP, FP, FN
        tps = np.diag(self.matrix)
        fps = np.sum(self.matrix, axis=0) - tps
        fns = np.sum(self.matrix, axis=1) - tps

        if metric == "precision":
            # 精确率 = TP / (TP + FP)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                precision_per_class = tps / (tps + fps)
                precision_per_class = np.nan_to_num(precision_per_class, nan=0.0)
        elif metric == "recall":
            # 召回率 = TP / (TP + FN)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                recall_per_class = tps / (tps + fns)
                recall_per_class = np.nan_to_num(recall_per_class, nan=0.0)
        elif metric == "f1":
            # F1 = 2 * (precision * recall) / (precision + recall)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                precision = tps / (tps + fps)
                recall = tps / (tps + fns)
                f1_per_class = 2 * (precision * recall) / (precision + recall)
                f1_per_class = np.nan_to_num(f1_per_class, nan=0.0)
        else:
            raise ValueError(f"未知的指标类型: {metric}")

        if average == "micro":
            # 微平均
            total_tp = np.sum(tps)
            total_fp = np.sum(fps)
            total_fn = np.sum(fns)

            if metric == "precision":
                return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            elif metric == "recall":
                return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            elif metric == "f1":
                precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        elif average == "macro":
            # 宏平均
            if metric == "f1":
                return np.mean(f1_per_class)
            elif metric == "precision":
                return np.mean(precision_per_class)
            elif metric == "recall":
                return np.mean(recall_per_class)

        elif average == "weighted":
            # 加权平均
            class_counts = np.sum(self.matrix, axis=1)
            weights = class_counts / np.sum(class_counts)

            if metric == "f1":
                return np.sum(weights * f1_per_class)
            elif metric == "precision":
                return np.sum(weights * precision_per_class)
            elif metric == "recall":
                return np.sum(weights * recall_per_class)

        else:
            raise ValueError(f"未知的平均方式: {average}")

    def get_classification_report(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取分类报告

        Args:
            class_names: 类别名称列表

        Returns:
            分类报告字典
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        report = {
            "class_names": class_names,
            "confusion_matrix": self.matrix.tolist(),
            "precision_macro": self.get_precision("macro"),
            "recall_macro": self.get_recall("macro"),
            "f1_macro": self.get_f1_score("macro"),
            "precision_weighted": self.get_precision("weighted"),
            "recall_weighted": self.get_recall("weighted"),
            "f1_weighted": self.get_f1_score("weighted"),
            "per_class": {}
        }

        # 计算每个类别的指标
        tps = np.diag(self.matrix)
        fps = np.sum(self.matrix, axis=0) - tps
        fns = np.sum(self.matrix, axis=1) - tps

        for i in range(self.num_classes):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                precision = tps[i] / (tps[i] + fps[i]) if (tps[i] + fps[i]) > 0 else 0.0
                recall = tps[i] / (tps[i] + fns[i]) if (tps[i] + fns[i]) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            report["per_class"][class_names[i]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(np.sum(self.matrix[i, :]))
            }

        return report

    def plot_confusion_matrix(self, class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> None:
        """
        绘制混淆矩阵

        Args:
            class_names: 类别名称列表
            save_path: 保存路径，如果为None则显示图像
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if class_names is None:
                class_names = [f"Class {i}" for i in range(self.num_classes)]

            fig, ax = plt.subplots(figsize=(10, 8))

            # 归一化混淆矩阵
            normalized_matrix = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
            normalized_matrix = np.nan_to_num(normalized_matrix)

            # 绘制热图
            sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': '比例'}, ax=ax)

            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('混淆矩阵')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"混淆矩阵保存到: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("警告: matplotlib或seaborn未安装，无法绘制混淆矩阵")


class MetricCollection:
    """指标集合"""

    def __init__(self, metrics: Optional[Dict[str, Metric]] = None):
        """
        初始化指标集合

        Args:
            metrics: 指标字典
        """
        self.metrics = metrics or {}

    def add_metric(self, name: str, metric: Metric) -> None:
        """
        添加指标

        Args:
            name: 指标名称
            metric: 指标对象
        """
        self.metrics[name] = metric

    def reset(self) -> None:
        """重置所有指标"""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, updates: Dict[str, Any]) -> None:
        """
        更新指标

        Args:
            updates: 更新字典，键为指标名称，值为更新值
        """
        for name, value in updates.items():
            if name in self.metrics:
                if isinstance(value, tuple) and len(value) == 2:
                    self.metrics[name].update(value[0], value[1])
                else:
                    self.metrics[name].update(value)

    def compute(self) -> Dict[str, Any]:
        """
        计算所有指标

        Returns:
            指标结果字典
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        return results

    def get_metric(self, name: str) -> Metric:
        """
        获取指标对象

        Args:
            name: 指标名称

        Returns:
            指标对象
        """
        return self.metrics[name]


def test_metrics():
    """测试指标模块"""
    print("测试指标模块...")

    # 测试AverageMetric
    avg_metric = AverageMetric("loss")
    avg_metric.update(0.5, 10)
    avg_metric.update(0.3, 5)
    print(f"AverageMetric: {avg_metric.compute():.4f} (期望: ~0.4333)")

    # 测试AccuracyMetric
    acc_metric = AccuracyMetric("accuracy")
    acc_metric.update(0.8, 10)
    acc_metric.update(0.9, 20)
    print(f"AccuracyMetric: {acc_metric.compute():.4f} (期望: 0.8667)")

    # 测试从预测结果更新
    predictions = torch.tensor([0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 1, 1])
    acc_metric.update_from_predictions(predictions, targets)
    print(f"Accuracy from predictions: {acc_metric.compute():.4f} (期望: ~0.8286)")

    # 测试ConfusionMatrixMetric
    cm_metric = ConfusionMatrixMetric(num_classes=3)
    cm_metric.reset()

    predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    targets = torch.tensor([0, 1, 2, 0, 1, 1, 0, 2])

    cm_metric.update(predictions, targets)
    matrix = cm_metric.compute()
    print(f"Confusion Matrix:\n{matrix}")

    print(f"Precision (macro): {cm_metric.get_precision('macro'):.4f}")
    print(f"Recall (macro): {cm_metric.get_recall('macro'):.4f}")
    print(f"F1 (macro): {cm_metric.get_f1_score('macro'):.4f}")

    # 测试分类报告
    class_names = ["A", "B", "C"]
    report = cm_metric.get_classification_report(class_names)
    print(f"Classification report keys: {list(report.keys())}")

    # 测试MetricCollection
    collection = MetricCollection({
        "loss": AverageMetric("loss"),
        "accuracy": AccuracyMetric("accuracy"),
    })

    collection.update({
        "loss": (0.5, 10),
        "accuracy": (0.8, 10)
    })

    results = collection.compute()
    print(f"MetricCollection results: {results}")

    print("\n指标模块测试通过!")


if __name__ == "__main__":
    test_metrics()