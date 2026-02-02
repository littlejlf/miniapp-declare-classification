# -*- coding: utf-8 -*-
"""
评估指标计算模块

提供分类任务的各种评估指标计算功能。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ClassificationMetrics:
    """分类指标计算类"""

    def __init__(self, labels: Optional[List[str]] = None):
        """
        初始化指标计算器

        Args:
            labels: 标签名称列表
        """
        self.labels = labels or ['necessary', 'unnecessary']

    def compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        计算各种分类指标

        Args:
            y_true: 真实标签列表
            y_pred: 预测标签列表
            average: 多分类时的平均方式 ('micro', 'macro', 'weighted', 'binary')

        Returns:
            包含各种指标的字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        return metrics

    def compute_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> np.ndarray:
        """
        计算混淆矩阵

        Args:
            y_true: 真实标签列表
            y_pred: 预测标签列表

        Returns:
            混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        获取分类报告

        Args:
            y_true: 真实标签列表
            y_pred: 预测标签列表
            target_names: 目标类别名称

        Returns:
            分类报告字符串
        """
        if target_names is None:
            target_names = self.labels

        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        )


class MultiLabelMetrics:
    """多标签分类指标计算类（用于同时评估必要性和清晰性）"""

    def __init__(self):
        """初始化多标签指标计算器"""
        self.necessity_metrics = ClassificationMetrics(labels=['necessary', 'unnecessary'])
        self.clarity_metrics = ClassificationMetrics(labels=['clear', 'ambiguous'])

    def compute_all_metrics(
        self,
        y_true_necessity: List[int],
        y_pred_necessity: List[int],
        y_true_clarity: List[int],
        y_pred_clarity: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        计算所有维度的指标

        Args:
            y_true_necessity: 必要性真实标签
            y_pred_necessity: 必要性预测标签
            y_true_clarity: 清晰性真实标签
            y_pred_clarity: 清晰性预测标签

        Returns:
            包含所有维度指标的嵌套字典
        """
        results = {
            'necessity': self.necessity_metrics.compute_metrics(
                y_true_necessity, y_pred_necessity
            ),
            'clarity': self.clarity_metrics.compute_metrics(
                y_true_clarity, y_pred_clarity
            )
        }

        # 计算整体准确率（两个维度都正确才认为正确）
        both_correct = [
            1 if tn == pn and tc == pc else 0
            for tn, pn, tc, pc in zip(
                y_true_necessity, y_pred_necessity,
                y_true_clarity, y_pred_clarity
            )
        ]
        results['overall_accuracy'] = {
            'both_correct': np.mean(both_correct)
        }

        return results


def compute_consistency(
    predictions_a: List[int],
    predictions_b: List[int]
) -> Dict[str, Any]:
    """
    计算两个模型预测的一致性

    Args:
        predictions_a: 模型A的预测
        predictions_b: 模型B的预测

    Returns:
        一致性统计信息
    """
    if len(predictions_a) != len(predictions_b):
        raise ValueError("两个预测列表长度必须相同")

    # 计算一致的数量
    agreements = [1 if a == b else 0 for a, b in zip(predictions_a, predictions_b)]

    # Cohen's Kappa (分类任务一致性评估)
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(predictions_a, predictions_b)

    return {
        'agreement_rate': np.mean(agreements),
        'disagreement_rate': 1 - np.mean(agreements),
        'cohen_kappa': kappa,
        'total_agreements': sum(agreements),
        'total_disagreements': len(agreements) - sum(agreements)
    }


def analyze_conflicts(
    predictions_dict: Dict[str, List[int]],
    labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    分析多个模型之间的预测冲突

    Args:
        predictions_dict: 模型名称到预测列表的字典
        labels: 真实标签（可选）

    Returns:
        冲突分析结果
    """
    model_names = list(predictions_dict.keys())
    num_samples = len(next(iter(predictions_dict.values())))

    # 统计每个样本的不同预测数量
    conflict_counts = []
    conflict_samples = []

    for i in range(num_samples):
        predictions_for_sample = [predictions_dict[model][i] for model in model_names]
        unique_predictions = set(predictions_for_sample)

        if len(unique_predictions) > 1:
            conflict_counts.append(len(unique_predictions))
            conflict_samples.append({
                'index': i,
                'predictions': {
                    model: predictions_dict[model][i]
                    for model in model_names
                },
                'unique_count': len(unique_predictions)
            })

    # 如果有真实标签，分析哪些模型的预测是正确的
    if labels is not None:
        for sample in conflict_samples:
            idx = sample['index']
            true_label = labels[idx]
            sample['true_label'] = true_label
            sample['correct_models'] = [
                model for model in model_names
                if predictions_dict[model][idx] == true_label
            ]

    return {
        'total_conflicts': len(conflict_samples),
        'conflict_rate': len(conflict_samples) / num_samples if num_samples > 0 else 0,
        'avg_unique_predictions': np.mean(conflict_counts) if conflict_counts else 0,
        'conflict_samples': conflict_samples
    }


def format_metrics_report(metrics_dict: Dict[str, float], title: str = "Metrics Report") -> str:
    """
    格式化指标报告

    Args:
        metrics_dict: 指标字典
        title: 报告标题

    Returns:
        格式化的报告字符串
    """
    lines = [f"\n{'='*50}", f"{title}", f"{'='*50}"]

    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            lines.append(f"{metric_name:20s}: {value:.4f}")
        elif isinstance(value, dict):
            lines.append(f"\n{metric_name}:")
            for sub_name, sub_value in value.items():
                lines.append(f"  {sub_name:18s}: {sub_value:.4f}")
        else:
            lines.append(f"{metric_name:20s}: {value}")

    lines.append('='*50)
    return '\n'.join(lines)
