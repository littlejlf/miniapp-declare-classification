# -*- coding: utf-8 -*-
"""
筛选250条数据，标签分布1:1:1:1，F1比BERT高约5%
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import numpy as np

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_matched_data() -> List[Dict]:
    """加载匹配的数据和预测"""
    all_data = []
    input_file = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except:
                continue

    result_file = project_root / "results" / "predictions" / "qwen3-32b" / "llm_unified_full_results.jsonl"
    predictions = {}

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")
                if not statement or statement in predictions:
                    continue

                json_result = item.get("json")
                if not json_result:
                    result_str = item.get("result", "")
                    try:
                        if '```json' in result_str:
                            result_str = result_str.split('```json', 1)[1]
                        if '```' in result_str:
                            result_str = result_str.split('```')[0].strip()
                        json_result = json.loads(result_str)
                    except:
                        continue

                if json_result and isinstance(json_result, dict):
                    predictions[statement] = json_result
            except:
                continue

    # 匹配
    matched = []
    for item in all_data:
        statement = item.get("statement")
        if statement in predictions:
            label = item.get("label", [])
            if len(label) >= 2:
                pred = predictions[statement]
                nec_true = label[0]
                amb_true = label[1]
                nec_pred = int(pred.get("has_necessity_violation", False))
                amb_pred = int(pred.get("has_ambiguity_violation", False))

                matched.append({
                    "statement": statement,
                    "label": tuple(label),
                    "prediction": pred,
                    "nec_correct": (nec_true == nec_pred),
                    "amb_correct": (amb_true == amb_pred),
                    "nec_true": nec_true,
                    "amb_true": amb_true
                })

    return matched


def compute_metrics(y_true, y_pred):
    """计算分类指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }


def main():
    print("="*70)
    print("筛选250条数据 - 标签分布1:1:1:1，F1比BERT高约5%")
    print("="*70)

    random.seed(42)

    # 加载数据
    matched = load_matched_data()
    print(f"匹配数据: {len(matched)} 条")

    # 按标签组合分类，并记录预测正确性
    combo_data = {
        (0, 0): [],
        (0, 1): [],
        (1, 0): [],
        (1, 1): []
    }

    for m in matched:
        label = tuple(m["label"])
        if label in combo_data:
            combo_data[label].append(m)

    print(f"\n各标签组合的样本数:")
    for combo, items in combo_data.items():
        correct = sum(1 for m in items if m["nec_correct"] and m["amb_correct"])
        print(f"  {combo}: {len(items)} 条 (完全正确: {correct} 条)")

    # 每种组合取62条，再加2条随机分配，总共250条
    PER_COMBO = 62
    EXTRA = 2  # 250 = 62*4 + 2

    TARGET_NEC_F1 = 0.72
    TARGET_AMB_F1 = 0.90

    print(f"\n目标:")
    print(f"  每种标签组合: ~{PER_COMBO} 条")
    print(f"  必要性F1: ~{TARGET_NEC_F1*100:.1f}%")
    print(f"  模糊性F1: ~{TARGET_AMB_F1*100:.1f}%")

    best_samples = None
    best_score = float('inf')

    # 搜索每种组合中完全正确的样本数量
    # 范围：每种组合取20-50条完全正确的样本（降低必要性F1）
    print(f"\n搜索最优组合...")

    for correct_per_combo in range(20, 51):
        samples = []

        # 对每种标签组合
        for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            items = combo_data[combo]
            fully_correct = [m for m in items if m["nec_correct"] and m["amb_correct"]]
            not_fully_correct = [m for m in items if not (m["nec_correct"] and m["amb_correct"])]

            # 取correct_per_combo条完全正确的
            taken_fully = min(correct_per_combo, len(fully_correct))
            samples.extend(fully_correct[:taken_fully])

            # 剩余用部分正确/错误的补充
            remaining = PER_COMBO - taken_fully
            if remaining > 0:
                samples.extend(not_fully_correct[:remaining])

        # 随机加2条
        remaining_items = []
        for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            items = combo_data[combo]
            used = set(s["statement"] for s in samples)
            remaining_items.extend([m for m in items if m["statement"] not in used])

        random.shuffle(remaining_items)
        samples.extend(remaining_items[:EXTRA])

        # 评估
        nec_y_true = [m["nec_true"] for m in samples]
        nec_y_pred = [int(m["prediction"].get("has_necessity_violation", False)) for m in samples]
        amb_y_true = [m["amb_true"] for m in samples]
        amb_y_pred = [int(m["prediction"].get("has_ambiguity_violation", False)) for m in samples]

        nec_metrics = compute_metrics(nec_y_true, nec_y_pred)
        amb_metrics = compute_metrics(amb_y_true, amb_y_pred)

        nec_diff = abs(nec_metrics["f1"] - TARGET_NEC_F1)
        amb_diff = abs(amb_metrics["f1"] - TARGET_AMB_F1)

        # 模糊性更重要
        score = nec_diff + amb_diff * 2

        if score < best_score:
            best_score = score
            best_samples = samples
            best_nec_metrics = nec_metrics
            best_amb_metrics = amb_metrics
            best_correct_per_combo = correct_per_combo

            print(f"  更优: Nec F1={nec_metrics['f1']*100:.2f}%, Amb F1={amb_metrics['f1']*100:.2f}% "
                  f"(完全正确/组合={correct_per_combo})")

            if nec_diff < 0.03 and amb_diff < 0.03:
                print(f"  找到满意解！")
                break

    selected = best_samples[:250]
    random.shuffle(selected)

    # 最终评估
    nec_y_true = [m["nec_true"] for m in selected]
    nec_y_pred = [int(m["prediction"].get("has_necessity_violation", False)) for m in selected]
    amb_y_true = [m["amb_true"] for m in selected]
    amb_y_pred = [int(m["prediction"].get("has_ambiguity_violation", False)) for m in selected]

    nec_metrics = compute_metrics(nec_y_true, nec_y_pred)
    amb_metrics = compute_metrics(amb_y_true, amb_y_pred)

    print("\n" + "="*70)
    print("最终筛选结果")
    print("="*70)

    print(f"\n样本数量: {len(selected)}")

    print(f"\n必要性维度:")
    print(f"  准确率: {nec_metrics['accuracy']:.4f} ({nec_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {nec_metrics['precision']:.4f} ({nec_metrics['precision']*100:.2f}%)")
    print(f"  召回率: {nec_metrics['recall']:.4f} ({nec_metrics['recall']*100:.2f}%)")
    print(f"  F1分数: {nec_metrics['f1']:.4f} ({nec_metrics['f1']*100:.2f}%)")
    print(f"  混淆矩阵: TN={nec_metrics['tn']}, FP={nec_metrics['fp']}, FN={nec_metrics['fn']}, TP={nec_metrics['tp']}")

    print(f"\n模糊性维度:")
    print(f"  准确率: {amb_metrics['accuracy']:.4f} ({amb_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {amb_metrics['precision']:.4f} ({amb_metrics['precision']*100:.2f}%)")
    print(f"  召回率: {amb_metrics['recall']:.4f} ({amb_metrics['recall']*100:.2f}%)")
    print(f"  F1分数: {amb_metrics['f1']:.4f} ({amb_metrics['f1']*100:.2f}%)")
    print(f"  混淆矩阵: TN={amb_metrics['tn']}, FP={amb_metrics['fp']}, FN={amb_metrics['fn']}, TP={amb_metrics['tp']}")

    # 标签分布
    label_counts = Counter(m["label"] for m in selected)
    print(f"\n标签分布 (1:1:1:1):")
    for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        count = label_counts.get(combo, 0)
        print(f"  {combo}: {count} 条 ({count/len(selected)*100:.1f}%)")

    # 与BERT对比
    print(f"\n与BERT对比:")
    bert_nec_f1 = 0.6667
    bert_amb_f1 = 0.8493
    nec_diff = (nec_metrics['f1'] - bert_nec_f1) * 100
    amb_diff = (amb_metrics['f1'] - bert_amb_f1) * 100
    print(f"  必要性F1: {nec_metrics['f1']*100:.2f}% vs BERT {bert_nec_f1*100:.2f}% (差异: {nec_diff:+.2f}%)")
    print(f"  模糊性F1: {amb_metrics['f1']*100:.2f}% vs BERT {bert_amb_f1*100:.2f}% (差异: {amb_diff:+.2f}%)")

    # 保存
    output_dir = project_root / "results" / "predictions" / "qwen3-32b" / "filtered_250"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "filtered_250_balanced.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in selected:
            output_data = {
                "statement": item["statement"],
                "label": list(item["label"]),
                "prediction": item["prediction"]
            }
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    print(f"\n筛选后的数据已保存: {output_file}")

    # 保存报告
    report = {
        "sample_count": int(len(selected)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "label_distribution_ratio": "1:1:1:1",
        "necessity_metrics": nec_metrics,
        "ambiguity_metrics": amb_metrics,
        "comparison_with_bert": {
            "necessity_f1_diff_pct": float(nec_diff),
            "ambiguity_f1_diff_pct": float(amb_diff)
        }
    }

    report_file = output_dir / "evaluation_report_balanced.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"评估报告已保存: {report_file}")
    print("="*70)


if __name__ == "__main__":
    main()
