# -*- coding: utf-8 -*-
"""
分类结果评估模块

支持评估三种分类器的结果：
1. 统一分类器结果
2. 独立分类器合并结果
3. 对比分析

评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵
- 分类报告
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.metrics import (
    ClassificationMetrics,
    MultiLabelMetrics,
    format_metrics_report,
    compute_consistency
)
from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)


def parse_llm_result(content: str) -> Dict:
    """解析LLM返回的JSON结果"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}


def read_jsonl(file_path: Path) -> List[Dict]:
    """读取JSONL文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return results


def read_labels(label_file: Path) -> Dict[str, List[int]]:
    """读取标签文件，返回 {statement: [necessity_label, ambiguity_label]}"""
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    stmt = data.get("statement")
                    label = data.get("label")
                    if stmt and label:
                        labels[stmt] = label
                except json.JSONDecodeError:
                    continue
    return labels


def evaluate_unified_results(
    result_file: Path,
    label_file: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """评估统一分类器结果"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估统一分类器结果")
    logger.info(f"{'='*60}")

    # 读取数据和标签
    results = read_jsonl(result_file)
    labels = read_labels(label_file)

    logger.info(f"结果数量: {len(results)}")
    logger.info(f"标签数量: {len(labels)}")

    # 提取预测和真实标签
    y_true_necessity = []
    y_pred_necessity = []
    y_true_ambiguity = []
    y_pred_ambiguity = []
    matched = []

    for item in results:
        stmt = item.get("statement")
        if stmt in labels:
            result_json = parse_llm_result(item.get("result", "{}"))
            true_label = labels[stmt]

            y_true_necessity.append(true_label[0])
            y_pred_necessity.append(int(result_json.get("has_necessity_violation", False)))
            y_true_ambiguity.append(true_label[1])
            y_pred_ambiguity.append(int(result_json.get("has_ambiguity_violation", False)))
            matched.append({
                "statement": stmt,
                "true_necessity": true_label[0],
                "pred_necessity": int(result_json.get("has_necessity_violation", False)),
                "true_ambiguity": true_label[1],
                "pred_ambiguity": int(result_json.get("has_ambiguity_violation", False)),
                "necessity_correct": true_label[0] == int(result_json.get("has_necessity_violation", False)),
                "ambiguity_correct": true_label[1] == int(result_json.get("has_ambiguity_violation", False))
            })

    logger.info(f"匹配数量: {len(matched)}")

    # 计算指标
    metrics = MultiLabelMetrics()
    all_metrics = metrics.compute_all_metrics(
        y_true_necessity, y_pred_necessity,
        y_true_ambiguity, y_pred_ambiguity
    )

    # 打印报告
    print(format_metrics_report(all_metrics["necessity"], "必要性违规指标"))
    print(format_metrics_report(all_metrics["ambiguity"], "表述模糊违规指标"))
    print(format_metrics_report(all_metrics["overall_accuracy"], "整体准确率"))

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "classifier": "unified",
            "total_samples": len(matched),
            "metrics": all_metrics,
            "detailed_results": matched
        }
        report_file = output_dir / "llm_prompt_unified_evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存到: {report_file}")

    return all_metrics


def evaluate_independent_results(
    result_file: Path,
    label_file: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """评估独立分类器合并结果"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估独立分类器结果")
    logger.info(f"{'='*60}")

    # 读取数据和标签
    results = read_jsonl(result_file)
    labels = read_labels(label_file)

    logger.info(f"结果数量: {len(results)}")
    logger.info(f"标签数量: {len(labels)}")

    # 提取预测和真实标签
    y_true_necessity = []
    y_pred_necessity = []
    y_true_ambiguity = []
    y_pred_ambiguity = []
    matched = []

    for item in results:
        stmt = item.get("statement")
        if stmt in labels:
            necessity = item.get("necessity", {})
            ambiguity = item.get("ambiguity", {})
            true_label = labels[stmt]

            y_true_necessity.append(true_label[0])
            y_pred_necessity.append(int(necessity.get("has_violation", False)))
            y_true_ambiguity.append(true_label[1])
            y_pred_ambiguity.append(int(ambiguity.get("has_violation", False)))
            matched.append({
                "statement": stmt,
                "true_necessity": true_label[0],
                "pred_necessity": int(necessity.get("has_violation", False)),
                "true_ambiguity": true_label[1],
                "pred_ambiguity": int(ambiguity.get("has_violation", False)),
                "necessity_correct": true_label[0] == int(necessity.get("has_violation", False)),
                "ambiguity_correct": true_label[1] == int(ambiguity.get("has_violation", False))
            })

    logger.info(f"匹配数量: {len(matched)}")

    # 计算指标
    metrics = MultiLabelMetrics()
    all_metrics = metrics.compute_all_metrics(
        y_true_necessity, y_pred_necessity,
        y_true_ambiguity, y_pred_ambiguity
    )

    # 打印报告
    print(format_metrics_report(all_metrics["necessity"], "必要性违规指标"))
    print(format_metrics_report(all_metrics["ambiguity"], "表述模糊违规指标"))
    print(format_metrics_report(all_metrics["overall_accuracy"], "整体准确率"))

    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "classifier": "independent",
            "total_samples": len(matched),
            "metrics": all_metrics,
            "detailed_results": matched
        }
        report_file = output_dir / "independent_evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存到: {report_file}")

    return all_metrics


def compare_classifiers(
    result_file: Path,
    label_file: Path,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """对比统一分类器和独立分类器的性能"""
    logger.info(f"\n{'='*60}")
    logger.info(f"对比分类器性能")
    logger.info(f"{'='*60}")

    # 读取标签
    labels = read_labels(label_file)

    # 读取统一分类器结果
    unified_results = read_jsonl(result_file.parent / "llm_unified_results.jsonl")
    unified_preds = {}
    for item in unified_results:
        stmt = item.get("statement")
        if stmt in labels:
            result_json = parse_llm_result(item.get("result", "{}"))
            unified_preds[stmt] = {
                "necessity": int(result_json.get("has_necessity_violation", False)),
                "ambiguity": int(result_json.get("has_ambiguity_violation", False))
            }

    # 读取独立分类器结果
    independent_results = read_jsonl(result_file)
    independent_preds = {}
    for item in independent_results:
        stmt = item.get("statement")
        if stmt in labels:
            necessity = item.get("necessity", {})
            ambiguity = item.get("ambiguity", {})
            independent_preds[stmt] = {
                "necessity": int(necessity.get("has_violation", False)),
                "ambiguity": int(ambiguity.get("has_violation", False))
            }

    # 找到共同的样本
    common_statements = set(unified_preds.keys()) & set(independent_preds.keys())

    # 提取预测
    unified_necessity_preds = []
    independent_necessity_preds = []
    unified_ambiguity_preds = []
    independent_ambiguity_preds = []
    true_necessity = []
    true_ambiguity = []

    for stmt in common_statements:
        true_label = labels[stmt]
        true_necessity.append(true_label[0])
        true_ambiguity.append(true_label[1])
        unified_necessity_preds.append(unified_preds[stmt]["necessity"])
        independent_necessity_preds.append(independent_preds[stmt]["necessity"])
        unified_ambiguity_preds.append(unified_preds[stmt]["ambiguity"])
        independent_ambiguity_preds.append(independent_preds[stmt]["ambiguity"])

    # 计算各自的准确率
    from sklearn.metrics import accuracy_score

    unified_necessity_acc = accuracy_score(true_necessity, unified_necessity_preds)
    independent_necessity_acc = accuracy_score(true_necessity, independent_necessity_preds)
    unified_ambiguity_acc = accuracy_score(true_ambiguity, unified_ambiguity_preds)
    independent_ambiguity_acc = accuracy_score(true_ambiguity, independent_ambiguity_preds)

    # 计算一致性
    necessity_consistency = compute_consistency(unified_necessity_preds, independent_necessity_preds)
    ambiguity_consistency = compute_consistency(unified_ambiguity_preds, independent_ambiguity_preds)

    # 打印对比结果
    print(f"\n{'='*60}")
    print(f"分类器性能对比")
    print(f"{'='*60}")
    print(f"\n必要性违规:")
    print(f"  统一分类器准确率:     {unified_necessity_acc:.4f}")
    print(f"  独立分类器准确率:     {independent_necessity_acc:.4f}")
    print(f"  两者一致性:           {necessity_consistency['agreement_rate']:.4f}")
    print(f"  Cohen's Kappa:        {necessity_consistency['cohen_kappa']:.4f}")

    print(f"\n表述模糊违规:")
    print(f"  统一分类器准确率:     {unified_ambiguity_acc:.4f}")
    print(f"  独立分类器准确率:     {independent_ambiguity_acc:.4f}")
    print(f"  两者一致性:           {ambiguity_consistency['agreement_rate']:.4f}")
    print(f"  Cohen's Kappa:        {ambiguity_consistency['cohen_kappa']:.4f}")
    print(f"{'='*60}\n")

    # 保存对比结果
    comparison = {
        "necessity": {
            "unified_accuracy": unified_necessity_acc,
            "independent_accuracy": independent_necessity_acc,
            "consistency": necessity_consistency
        },
        "ambiguity": {
            "unified_accuracy": unified_ambiguity_acc,
            "independent_accuracy": independent_ambiguity_acc,
            "consistency": ambiguity_consistency
        }
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "classifier_comparison_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f"对比报告已保存到: {report_file}")

    return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估分类结果")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入结果文件路径"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="标签文件路径"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["unified", "independent", "both", "compare"],
        default="both",
        help="评估类型: unified=统一分类器, independent=独立分类器, both=两者都评估, compare=对比分析"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认保存在结果文件同级的evaluation目录）"
    )

    args = parser.parse_args()

    # 设置输出目录
    input_path = Path(args.input)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / "evaluation"

    # 根据类型执行评估
    if args.type in ["unified", "both"]:
        evaluate_unified_results(input_path, Path(args.labels), output_dir)

    if args.type in ["independent", "both"]:
        evaluate_independent_results(input_path, Path(args.labels), output_dir)

    if args.type == "compare":
        compare_classifiers(input_path, Path(args.labels), output_dir)


if __name__ == "__main__":
    main()
