# -*- coding: utf-8 -*-
"""
LLM分类结果评估脚本

功能：
1. 加载实验结果和真实标签
2. 计算准确率、精确率、召回率、F1分数
3. 生成混淆矩阵
4. 比较统一分类器和独立分类器的性能

使用方法:
    python evaluate_results.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
MODEL_ID = "qwen3-14b"
RESULT_DIR = project_root / "results" / "predictions" / MODEL_ID

# ==================== 工具函数 ====================
def load_ground_truth() -> Dict[str, Tuple[int, int]]:
    """
    加载真实标签

    Returns:
        Dict[str, Tuple[int, int)]: statement -> (necessity_label, ambiguity_label)
    """
    sampled_file = RESULT_DIR / "sampled_200_list.json"

    with open(sampled_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ground_truth = {}
    for item in data:
        statement = item["statement"]
        label = item["label"]  # [necessity, ambiguity]
        ground_truth[statement] = (label[0], label[1])

    return ground_truth


def load_predictions(result_file: Path) -> Dict[str, Dict]:
    """
    加载预测结果

    Args:
        result_file: 结果文件路径

    Returns:
        Dict[str, Dict]: statement -> prediction dict
    """
    predictions = {}

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")

                # 只处理成功的预测
                if item.get("status") == "success" and statement:
                    # 如果已经存在，跳过重复的
                    if statement not in predictions:
                        # 解析JSON结果
                        json_result = item.get("json")
                        if json_result:
                            predictions[statement] = json_result
                        else:
                            # 尝试解析result字段
                            try:
                                result_str = item.get("result", "")
                                if result_str:
                                    json_result = json.loads(result_str)
                                    predictions[statement] = json_result
                            except:
                                pass
            except json.JSONDecodeError:
                continue

    return predictions


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    计算分类指标

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        包含accuracy, precision, recall, f1的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 准确率
    accuracy = np.mean(y_true == y_pred)

    # 计算TP, TN, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # 精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    计算混淆矩阵

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        2x2混淆矩阵 [[TN, FP], [FN, TP]]
    """
    cm = np.zeros((2, 2), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    return cm


def evaluate_classifier(
    ground_truth: Dict[str, Tuple[int, int]],
    predictions: Dict[str, Dict],
    dimension: str  # "necessity" or "ambiguity"
) -> Dict:
    """
    评估单个分类器

    Args:
        ground_truth: 真实标签字典
        predictions: 预测结果字典
        dimension: 评估维度

    Returns:
        评估指标字典
    """
    y_true = []
    y_pred = []
    matched = 0
    missing = 0

    for statement, (nec_true, amb_true) in ground_truth.items():
        true_label = nec_true if dimension == "necessity" else amb_true

        if statement in predictions:
            pred = predictions[statement]
            pred_label = int(pred.get(f"has_{dimension}_violation", False))

            y_true.append(true_label)
            y_pred.append(pred_label)
            matched += 1
        else:
            missing += 1

    if len(y_true) == 0:
        return {"error": "No valid predictions"}

    metrics = compute_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        **metrics,
        "confusion_matrix": cm.tolist(),
        "matched": matched,
        "missing": missing,
        "total": len(ground_truth)
    }


def evaluate_unified_classifier(
    ground_truth: Dict[str, Tuple[int, int]],
    predictions: Dict[str, Dict]
) -> Dict:
    """
    评估统一分类器（两个维度同时正确）

    Args:
        ground_truth: 真实标签字典
        predictions: 预测结果字典

    Returns:
        评估指标字典
    """
    exact_match = 0
    partial_match = 0  # 至少一个维度正确
    total = 0

    for statement, (nec_true, amb_true) in ground_truth.items():
        if statement not in predictions:
            continue

        total += 1
        pred = predictions[statement]
        nec_pred = int(pred.get("has_necessity_violation", False))
        amb_pred = int(pred.get("has_ambiguity_violation", False))

        if nec_pred == nec_true and amb_pred == amb_true:
            exact_match += 1

        if nec_pred == nec_true or amb_pred == amb_true:
            partial_match += 1

    accuracy = exact_match / total if total > 0 else 0.0
    partial_accuracy = partial_match / total if total > 0 else 0.0

    return {
        "exact_accuracy": accuracy,
        "partial_accuracy": partial_accuracy,
        "exact_match": exact_match,
        "partial_match": partial_match,
        "total": total
    }


def print_metrics(metrics: Dict, title: str = ""):
    """打印评估指标"""
    print(f"\n{'='*60}")
    if title:
        print(f"{title}")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"错误: {metrics['error']}")
        return

    print(f"准确率 (Accuracy):  {metrics.get('accuracy', 0):.4f}")
    print(f"精确率 (Precision): {metrics.get('precision', 0):.4f}")
    print(f"召回率 (Recall):    {metrics.get('recall', 0):.4f}")
    print(f"F1分数 (F1-Score):  {metrics.get('f1', 0):.4f}")

    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print(f"\n混淆矩阵:")
        print(f"              预测: 无违规  预测: 有违规")
        print(f"实际: 无违规   {cm[0][0]:5d}      {cm[0][1]:5d}")
        print(f"实际: 有违规   {cm[1][0]:5d}      {cm[1][1]:5d}")

    print(f"\n匹配数量: {metrics.get('matched', metrics.get('total', 0))}/{metrics.get('total', 0)}")


def print_unified_metrics(metrics: Dict):
    """打印统一分类器指标"""
    print(f"\n{'='*60}")
    print(f"统一分类器 - 完全匹配评估")
    print(f"{'='*60}")
    print(f"完全匹配准确率: {metrics['exact_accuracy']:.4f}")
    print(f"部分匹配准确率: {metrics['partial_accuracy']:.4f}")
    print(f"完全匹配数量: {metrics['exact_match']}/{metrics['total']}")


# ==================== 主程序 ====================
def main():
    print("="*60)
    print("LLM分类结果评估")
    print("="*60)
    print(f"模型: {MODEL_ID}")
    print(f"结果目录: {RESULT_DIR}")
    print("="*60)

    # 加载真实标签
    print("\n加载真实标签...")
    ground_truth = load_ground_truth()
    print(f"加载了 {len(ground_truth)} 条真实标签")

    # 统计标签分布
    nec_labels = [label[0] for label in ground_truth.values()]
    amb_labels = [label[1] for label in ground_truth.values()]

    print(f"\n标签分布:")
    print(f"  必要性违规: {sum(nec_labels)} / {len(nec_labels)} ({sum(nec_labels)/len(nec_labels)*100:.1f}%)")
    print(f"  模糊性违规: {sum(amb_labels)} / {len(amb_labels)} ({sum(amb_labels)/len(amb_labels)*100:.1f}%)")

    # 加载统一分类器结果
    print(f"\n加载统一分类器结果...")
    unified_predictions = load_predictions(RESULT_DIR / "llm_unified_results.jsonl")
    print(f"  加载了 {len(unified_predictions)} 条预测结果")

    # 评估统一分类器
    print("\n" + "="*60)
    print("统一分类器评估")
    print("="*60)

    unified_nec = evaluate_classifier(ground_truth, unified_predictions, "necessity")
    print_metrics(unified_nec, "必要性维度")

    unified_amb = evaluate_classifier(ground_truth, unified_predictions, "ambiguity")
    print_metrics(unified_amb, "模糊性维度")

    unified_exact = evaluate_unified_classifier(ground_truth, unified_predictions)
    print_unified_metrics(unified_exact)

    # 加载独立分类器结果
    print(f"\n加载必要性分类器结果...")
    necessity_predictions = load_predictions(RESULT_DIR / "llm_necessity_results.jsonl")
    print(f"  加载了 {len(necessity_predictions)} 条预测结果")

    print(f"\n加载模糊性分类器结果...")
    ambiguity_predictions = load_predictions(RESULT_DIR / "llm_ambiguity_results.jsonl")
    print(f"  加载了 {len(ambiguity_predictions)} 条预测结果")

    # 评估独立分类器
    print("\n" + "="*60)
    print("独立分类器评估")
    print("="*60)

    nec_metrics = evaluate_classifier(ground_truth, necessity_predictions, "necessity")
    print_metrics(nec_metrics, "必要性分类器")

    amb_metrics = evaluate_classifier(ground_truth, ambiguity_predictions, "ambiguity")
    print_metrics(amb_metrics, "模糊性分类器")

    # 比较统一 vs 独立
    print("\n" + "="*60)
    print("统一分类器 vs 独立分类器对比")
    print("="*60)

    print(f"\n必要性维度:")
    print(f"  统一分类器 - 准确率: {unified_nec['accuracy']:.4f}, F1: {unified_nec['f1']:.4f}")
    print(f"  必要性分类器 - 准确率: {nec_metrics['accuracy']:.4f}, F1: {nec_metrics['f1']:.4f}")
    print(f"  差异: {(nec_metrics['accuracy'] - unified_nec['accuracy']):.4f}")

    print(f"\n模糊性维度:")
    print(f"  统一分类器 - 准确率: {unified_amb['accuracy']:.4f}, F1: {unified_amb['f1']:.4f}")
    print(f"  模糊性分类器 - 准确率: {amb_metrics['accuracy']:.4f}, F1: {amb_metrics['f1']:.4f}")
    print(f"  差异: {(amb_metrics['accuracy'] - unified_amb['accuracy']):.4f}")

    # 保存评估报告
    report = {
        "model_id": MODEL_ID,
        "total_samples": len(ground_truth),
        "label_distribution": {
            "necessity_violations": sum(nec_labels),
            "necessity_total": len(nec_labels),
            "ambiguity_violations": sum(amb_labels),
            "ambiguity_total": len(amb_labels)
        },
        "unified_classifier": {
            "necessity": {
                "accuracy": unified_nec.get("accuracy", 0),
                "precision": unified_nec.get("precision", 0),
                "recall": unified_nec.get("recall", 0),
                "f1": unified_nec.get("f1", 0),
                "confusion_matrix": unified_nec.get("confusion_matrix", []),
                "matched": unified_nec.get("matched", 0)
            },
            "ambiguity": {
                "accuracy": unified_amb.get("accuracy", 0),
                "precision": unified_amb.get("precision", 0),
                "recall": unified_amb.get("recall", 0),
                "f1": unified_amb.get("f1", 0),
                "confusion_matrix": unified_amb.get("confusion_matrix", []),
                "matched": unified_amb.get("matched", 0)
            },
            "exact_match_accuracy": unified_exact.get("exact_accuracy", 0),
            "partial_match_accuracy": unified_exact.get("partial_accuracy", 0)
        },
        "independent_classifiers": {
            "necessity": {
                "accuracy": nec_metrics.get("accuracy", 0),
                "precision": nec_metrics.get("precision", 0),
                "recall": nec_metrics.get("recall", 0),
                "f1": nec_metrics.get("f1", 0),
                "confusion_matrix": nec_metrics.get("confusion_matrix", []),
                "matched": nec_metrics.get("matched", 0)
            },
            "ambiguity": {
                "accuracy": amb_metrics.get("accuracy", 0),
                "precision": amb_metrics.get("precision", 0),
                "recall": amb_metrics.get("recall", 0),
                "f1": amb_metrics.get("f1", 0),
                "confusion_matrix": amb_metrics.get("confusion_matrix", []),
                "matched": amb_metrics.get("matched", 0)
            }
        }
    }

    report_file = RESULT_DIR / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"评估报告已保存到: {report_file}")
    print("="*60)

    return report


if __name__ == "__main__":
    report = main()
