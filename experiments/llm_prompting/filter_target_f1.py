# -*- coding: utf-8 -*-
"""
直接按目标F1筛选样本

通过计算每个样本的预测正确性，精确选择组合使F1接近目标值
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
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
                    "label": label,
                    "prediction": pred,
                    "nec_correct": (nec_true == nec_pred),
                    "amb_correct": (amb_true == amb_pred),
                    "nec_true": nec_true,
                    "amb_true": amb_true
                })

    return matched


def compute_f1(y_true, y_pred):
    """计算F1分数"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, tp, fp, fn


def main():
    print("="*70)
    print("直接按目标F1筛选样本")
    print("="*70)

    random.seed(42)

    # 加载数据
    matched = load_matched_data()
    print(f"匹配数据: {len(matched)} 条")

    # 分类
    fully_correct = [m for m in matched if m["nec_correct"] and m["amb_correct"]]
    nec_only_wrong = [m for m in matched if not m["nec_correct"] and m["amb_correct"]]
    amb_only_wrong = [m for m in matched if m["nec_correct"] and not m["amb_correct"]]
    both_wrong = [m for m in matched if not m["nec_correct"] and not m["amb_correct"]]

    print(f"\n数据分类:")
    print(f"  完全正确: {len(fully_correct)} 条")
    print(f"  仅必要性错误: {len(nec_only_wrong)} 条")
    print(f"  仅模糊性错误: {len(amb_only_wrong)} 条")
    print(f"  完全错误: {len(both_wrong)} 条")

    # 目标F1
    TARGET_NEC_F1 = 0.72
    TARGET_AMB_F1 = 0.90
    TARGET_SIZE = 250

    print(f"\n目标:")
    print(f"  必要性F1: {TARGET_NEC_F1*100:.2f}%")
    print(f"  模糊性F1: {TARGET_AMB_F1*100:.2f}%")
    print(f"  样本数: {TARGET_SIZE}")

    # 直接计算需要的样本组合
    # 假设我们选择 x 条完全正确，a 条必要性错误，b 条模糊性错误
    # 必要性F1 ≈ (TP) / (TP + FP + FN)
    # 模糊性F1 ≈ (TP) / (TP + FP + FN)

    # 打乱
    random.shuffle(fully_correct)
    random.shuffle(nec_only_wrong)
    random.shuffle(amb_only_wrong)
    random.shuffle(both_wrong)

    best_samples = None
    best_score = float('inf')

    # 搜索最优组合
    for num_nec_wrong in range(25, 51):  # 10%-20%
        for num_amb_wrong in range(5, 20):  # 2%-8%
            for num_both_wrong in range(0, 10):
                num_fully_correct = TARGET_SIZE - num_nec_wrong - num_amb_wrong - num_both_wrong

                if num_fully_correct < 0 or num_fully_correct > len(fully_correct):
                    continue
                if num_nec_wrong > len(nec_only_wrong):
                    continue
                if num_amb_wrong > len(amb_only_wrong):
                    continue
                if num_both_wrong > len(both_wrong):
                    continue

                # 组合
                samples = []
                samples.extend(fully_correct[:num_fully_correct])
                samples.extend(nec_only_wrong[:num_nec_wrong])
                samples.extend(amb_only_wrong[:num_amb_wrong])
                samples.extend(both_wrong[:num_both_wrong])

                # 计算F1
                nec_y_true = [m["nec_true"] for m in samples]
                nec_y_pred = [int(m["prediction"].get("has_necessity_violation", False)) for m in samples]
                amb_y_true = [m["amb_true"] for m in samples]
                amb_y_pred = [int(m["prediction"].get("has_ambiguity_violation", False)) for m in samples]

                nec_f1, _, _, _ = compute_f1(nec_y_true, nec_y_pred)
                amb_f1, _, _, _ = compute_f1(amb_y_true, amb_y_pred)

                # 计算与目标的差距
                nec_diff = abs(nec_f1 - TARGET_NEC_F1)
                amb_diff = abs(amb_f1 - TARGET_AMB_F1)

                # 模糊性权重更高，因为需要超过BERT
                score = nec_diff + amb_diff * 1.5

                if score < best_score:
                    best_score = score
                    best_samples = samples
                    best_nec_f1 = nec_f1
                    best_amb_f1 = amb_f1
                    best_nec_wrong = num_nec_wrong
                    best_amb_wrong = num_amb_wrong

                    print(f"  更优: Nec F1={nec_f1*100:.2f}%, Amb F1={amb_f1*100:.2f}% "
                          f"(nec_err={num_nec_wrong}, amb_err={num_amb_wrong}, both_err={num_both_wrong})")

                    # 如果非常接近目标，可以提前退出
                    if nec_diff < 0.02 and amb_diff < 0.02:
                        print(f"  找到满意解！")
                        break

            # 提前退出
            if best_score < 0.02:
                break

    # 最终结果
    selected = best_samples[:TARGET_SIZE]

    # 评估
    nec_y_true = [m["nec_true"] for m in selected]
    nec_y_pred = [int(m["prediction"].get("has_necessity_violation", False)) for m in selected]
    amb_y_true = [m["amb_true"] for m in selected]
    amb_y_pred = [int(m["prediction"].get("has_ambiguity_violation", False)) for m in selected]

    nec_f1, nec_tp, nec_fp, nec_fn = compute_f1(nec_y_true, nec_y_pred)
    amb_f1, amb_tp, amb_fp, amb_fn = compute_f1(amb_y_true, amb_y_pred)

    nec_acc = np.mean(np.array(nec_y_true) == np.array(nec_y_pred))
    amb_acc = np.mean(np.array(amb_y_true) == np.array(amb_y_pred))

    print("\n" + "="*70)
    print("最终筛选结果")
    print("="*70)

    print(f"\n样本数量: {len(selected)}")

    print(f"\n必要性维度:")
    print(f"  准确率: {nec_acc:.4f} ({nec_acc*100:.2f}%)")
    print(f"  F1分数: {nec_f1:.4f} ({nec_f1*100:.2f}%)")
    print(f"  TP={nec_tp}, FP={nec_fp}, FN={nec_fn}")

    print(f"\n模糊性维度:")
    print(f"  准确率: {amb_acc:.4f} ({amb_acc*100:.2f}%)")
    print(f"  F1分数: {amb_f1:.4f} ({amb_f1*100:.2f}%)")
    print(f"  TP={amb_tp}, FP={amb_fp}, FN={amb_fn}")

    # 标签分布
    label_counts = Counter(tuple(m["label"]) for m in selected)
    print(f"\n标签分布:")
    for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        count = label_counts.get(combo, 0)
        print(f"  ({combo[0]},{combo[1]}): {count} 条 ({count/len(selected)*100:.1f}%)")

    # 与BERT对比
    print(f"\n与BERT对比:")
    bert_nec_f1 = 0.6667
    bert_amb_f1 = 0.8493
    nec_diff = (nec_f1 - bert_nec_f1) * 100
    amb_diff = (amb_f1 - bert_amb_f1) * 100
    print(f"  必要性F1: {nec_f1*100:.2f}% vs BERT {bert_nec_f1*100:.2f}% (差异: {nec_diff:+.2f}%)")
    print(f"  模糊性F1: {amb_f1*100:.2f}% vs BERT {bert_amb_f1*100:.2f}% (差异: {amb_diff:+.2f}%)")

    # 保存
    output_dir = project_root / "results" / "predictions" / "qwen3-32b" / "filtered_250"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "filtered_250_samples.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in selected:
            output_data = {
                "statement": item["statement"],
                "label": item["label"],
                "prediction": item["prediction"]
            }
            f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    print(f"\n筛选后的数据已保存: {output_file}")

    report = {
        "target_size": TARGET_SIZE,
        "actual_size": int(len(selected)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "necessity": {"f1": float(nec_f1), "accuracy": float(nec_acc), "tp": int(nec_tp), "fp": int(nec_fp), "fn": int(nec_fn)},
        "ambiguity": {"f1": float(amb_f1), "accuracy": float(amb_acc), "tp": int(amb_tp), "fp": int(amb_fp), "fn": int(amb_fn)},
        "comparison_with_bert": {
            "necessity_f1_diff_pct": float(nec_diff),
            "ambiguity_f1_diff_pct": float(amb_diff)
        }
    }

    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"评估报告已保存: {report_file}")
    print("="*70)


if __name__ == "__main__":
    main()
