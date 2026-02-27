# -*- coding: utf-8 -*-
"""
测试结果评估脚本

评估LLM分类器在测试数据上的性能，包括：
- 准确率、精确率、召回率、F1分数
- 混淆矩阵
- 详细错误分析
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================== 配置 ====================
# API Key
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 路径配置
INPUT_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
TEST_DIR = project_root / "results" / "test"


# ==================== 工具函数 ====================
def load_labels(size: int = 10) -> Dict[str, Tuple[int, int]]:
    """加载标签数据 {statement: (necessity_label, ambiguity_label)}"""
    labels = {}
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= size:
                break
            try:
                item = json.loads(line.strip())
                stmt = item.get("statement")
                label = item.get("label")  # [necessity, ambiguity]
                if stmt and label:
                    labels[stmt] = tuple(label)
            except json.JSONDecodeError:
                continue
    return labels


def load_predictions(result_file: Path) -> Dict[str, Tuple[int, int]]:
    """加载预测结果 {statement: (necessity_pred, ambiguity_pred)}"""
    predictions = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                stmt = item.get("statement")
                result_json = item.get("json", {})

                if stmt and result_json:
                    nec = int(result_json.get("has_necessity_violation", False))
                    amb = int(result_json.get("has_ambiguity_violation", False))
                    predictions[stmt] = (nec, amb)
            except (json.JSONDecodeError, TypeError):
                continue
    return predictions


def match_data(labels: Dict, predictions: Dict) -> Tuple[List, List, List, List]:
    """匹配标签和预测，返回四个列表"""
    y_true_nec = []
    y_pred_nec = []
    y_true_amb = []
    y_pred_amb = []

    for stmt, label in labels.items():
        if stmt in predictions:
            pred = predictions[stmt]
            y_true_nec.append(label[0])
            y_pred_nec.append(pred[0])
            y_true_amb.append(label[1])
            y_pred_amb.append(pred[1])

    return y_true_nec, y_pred_nec, y_true_amb, y_pred_amb


# ==================== 评估函数 ====================
def evaluate_single_dimension(y_true, y_pred, dimension_name: str):
    """评估单个维度"""
    print(f"\n{'='*70}")
    print(f"{dimension_name} 分类评估")
    print(f"{'='*70}\n")

    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:")
    print(f"                预测正常    预测违规")
    print(f"实际正常:        {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"实际违规:        {cm[1][0]:>6}        {cm[1][1]:>6}")

    # 详细报告
    print(f"\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['正常/清晰', '违规/模糊'], zero_division=0))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }


def analyze_errors(y_true_nec, y_pred_nec, y_true_amb, y_pred_amb, labels, predictions):
    """分析错误样本"""
    print(f"\n{'='*70}")
    print("错误样本分析")
    print(f"{'='*70}\n")

    errors = []

    for stmt, true_label in labels.items():
        if stmt not in predictions:
            continue

        pred_label = predictions[stmt]

        nec_error = true_label[0] != pred_label[0]
        amb_error = true_label[1] != pred_label[1]

        if nec_error or amb_error:
            errors.append({
                "statement": stmt,
                "true": true_label,
                "pred": pred_label,
                "nec_error": nec_error,
                "amb_error": amb_error
            })

    if not errors:
        print("✓ 没有错误！所有预测都正确！")
        return

    print(f"错误数量: {len(errors)}\n")

    for idx, error in enumerate(errors, 1):
        stmt = error["statement"][:70]
        true_nec = "违规" if error["true"][0] else "正常"
        pred_nec = "违规" if error["pred"][0] else "正常"
        true_amb = "模糊" if error["true"][1] else "清晰"
        pred_amb = "模糊" if error["pred"][1] else "清晰"

        print(f"[{idx}] {stmt}...")
        if error["nec_error"]:
            print(f"    必要性: 实际={true_nec}, 预测={pred_nec} ❌")
        if error["amb_error"]:
            print(f"    表述: 实际={true_amb}, 预测={pred_amb} ❌")
        print()


# ==================== 主程序 ====================
def main():
    print("="*70)
    print("LLM分类器测试结果评估")
    print("="*70)
    print(f"数据源: {INPUT_FILE}")
    print(f"测试数量: 10")
    print("="*70)

    # 加载数据
    print("\n加载标签...")
    labels = load_labels(10)
    print(f"加载了 {len(labels)} 条标签")

    # 评估统一分类器
    print("\n加载统一分类器预测...")
    unified_predictions = load_predictions(TEST_DIR / "test_unified_results.jsonl")
    print(f"加载了 {len(unified_predictions)} 条预测")

    y_true_nec, y_pred_nec, y_true_amb, y_pred_amb = match_data(labels, unified_predictions)

    print(f"匹配成功: {len(y_true_nec)} 条")

    # 评估必要性
    nec_metrics = evaluate_single_dimension(y_true_nec, y_pred_nec, "必要性违规")

    # 评估表述模糊
    amb_metrics = evaluate_single_dimension(y_true_amb, y_pred_amb, "表述模糊违规")

    # 错误分析
    analyze_errors(y_true_nec, y_pred_nec, y_true_amb, y_pred_amb, labels, unified_predictions)

    # 保存评估结果
    print(f"\n{'='*70}")
    print("保存评估结果...")

    evaluation = {
        "test_size": 10,
        "matched": len(y_true_nec),
        "necessity_metrics": nec_metrics,
        "ambiguity_metrics": amb_metrics
    }

    output_file = TEST_DIR / "evaluation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"评估报告已保存: {output_file}")

    print(f"\n{'='*70}")
    print("评估完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
