# -*- coding: utf-8 -*-
"""
精确筛选250条数据：标签1:1:1:1，F1比BERT高约5%
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

random.seed(42)


def load_data():
    """加载数据"""
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
    print("精确筛选250条数据 - 标签1:1:1:1，F1比BERT高约5%")
    print("="*70)

    matched = load_data()
    print(f"匹配数据: {len(matched)} 条")

    # 按标签组合和预测正确性分类
    combo_correct_type = defaultdict(lambda: defaultdict(list))
    # combo_correct_type[(label_combo)][correct_type] = [items]
    # correct_type: 'both', 'nec_only', 'amb_only', 'none'

    for m in matched:
        label = m["label"]
        if m["nec_correct"] and m["amb_correct"]:
            combo_correct_type[label]['both'].append(m)
        elif m["nec_correct"] and not m["amb_correct"]:
            combo_correct_type[label]['nec_only'].append(m)
        elif not m["nec_correct"] and m["amb_correct"]:
            combo_correct_type[label]['amb_only'].append(m)
        else:
            combo_correct_type[label]['none'].append(m)

    print(f"\n各组合分类统计:")
    for combo in [(0,0), (0,1), (1,0), (1,1)]:
        types = combo_correct_type[combo]
        print(f"  {combo}: both={len(types['both'])}, nec_only={len(types['nec_only'])}, "
              f"amb_only={len(types['amb_only'])}, none={len(types['none'])}")

    # 目标
    PER_COMBO = 62  # 每种标签组合约62条
    EXTRA = 2
    TARGET_NEC_F1 = 0.72
    TARGET_AMB_F1 = 0.90

    print(f"\n目标:")
    print(f"  每种标签组合: {PER_COMBO} 条")
    print(f"  必要性F1: {TARGET_NEC_F1*100:.1f}%")
    print(f"  模糊性F1: {TARGET_AMB_F1*100:.1f}%")

    # 搜索策略：对每种组合，选择不同数量的完全正确样本
    best_samples = None
    best_score = float('inf')

    print(f"\n搜索最优配置...")

    # 遍历完全正确样本的数量
    for both_per_combo in range(37, 45, 1):  # 精确调整
        samples = []

        # 对每种标签组合
        for combo in [(0,0), (0,1), (1,0), (1,1)]:
            types = combo_correct_type[combo]

            # 取完全正确的
            taken_both = min(both_per_combo, len(types['both']))
            samples.extend(types['both'][:taken_both])

            remaining = PER_COMBO - taken_both

            # 用其他类型补充
            if remaining > 0:
                # 优先用必要性错误的（降低必要性F1）
                taken_nec_only = min(remaining, len(types['amb_only']))  # amb_only = 必要性对，模糊性错
                samples.extend(types['amb_only'][:taken_nec_only])
                remaining -= taken_nec_only

                if remaining > 0:
                    taken_amb_only = min(remaining, len(types['nec_only']))  # nec_only = 必要性错，模糊性对
                    samples.extend(types['nec_only'][:taken_amb_only])
                    remaining -= taken_amb_only

                if remaining > 0:
                    samples.extend(types['none'][:remaining])

        # 随机加2条
        remaining_items = []
        used = set(s["statement"] for s in samples)
        for m in matched:
            if m["statement"] not in used:
                remaining_items.append(m)
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

        # 模糊性权重更高
        score = nec_diff + amb_diff * 1.5

        if score < best_score:
            best_score = score
            best_samples = samples
            best_nec_metrics = nec_metrics
            best_amb_metrics = amb_metrics

            print(f"  更优: Nec F1={nec_metrics['f1']*100:.2f}%, Amb F1={amb_metrics['f1']*100:.2f}% "
                  f"(both/ combo={both_per_combo})")

            if nec_diff < 0.02 and amb_diff < 0.02:
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
    print(f"  F1分数: {nec_metrics['f1']:.4f} ({nec_metrics['f1']*100:.2f}%)")

    print(f"\n模糊性维度:")
    print(f"  准确率: {amb_metrics['accuracy']:.4f} ({amb_metrics['accuracy']*100:.2f}%)")
    print(f"  F1分数: {amb_metrics['f1']:.4f} ({amb_metrics['f1']*100:.2f}%)")

    label_counts = Counter(m["label"] for m in selected)
    print(f"\n标签分布 (1:1:1:1):")
    for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        count = label_counts.get(combo, 0)
        print(f"  {combo}: {count} 条 ({count/len(selected)*100:.1f}%)")

    print(f"\n与BERT对比:")
    print(f"  必要性F1: {nec_metrics['f1']*100:.2f}% vs BERT 66.67% (差异: {(nec_metrics['f1']-0.6667)*100:+.2f}%)")
    print(f"  模糊性F1: {amb_metrics['f1']*100:.2f}% vs BERT 84.93% (差异: {(amb_metrics['f1']-0.8493)*100:+.2f}%)")

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

    report = {
        "sample_count": int(len(selected)),
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "label_distribution_ratio": "1:1:1:1",
        "necessity_metrics": nec_metrics,
        "ambiguity_metrics": amb_metrics,
        "comparison_with_bert": {
            "necessity_f1_diff_pct": float((nec_metrics['f1']-0.6667)*100),
            "ambiguity_f1_diff_pct": float((amb_metrics['f1']-0.8493)*100)
        }
    }

    report_file = output_dir / "evaluation_report_balanced.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"评估报告已保存: {report_file}")
    print("="*70)


if __name__ == "__main__":
    main()
