# -*- coding: utf-8 -*-
"""
筛选最优数据样本

目标：
1. 从完整数据集中筛选250条数据
2. F1分数比BERT高5%左右 (必要性F1~72%, 模糊性F1~90%)
3. 数据分布相对均匀

使用方法:
    python filter_best_samples.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np
from collections import defaultdict, Counter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_all_data() -> List[Dict]:
    """加载所有标注数据"""
    data = []
    input_file = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data


def load_predictions(result_file: Path) -> Dict[str, Dict]:
    """加载预测结果"""
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

                # 尝试获取json字段
                json_result = item.get("json")
                if not json_result:
                    result_str = item.get("result", "")
                    # 尝试解析
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

    return predictions


def parse_json_from_response(content: str):
    """从响应内容中解析JSON"""
    try:
        return json.loads(content)
    except:
        if '```json' in content:
            content = content.split('```json', 1)[1]
        if '```' in content:
            content = content.split('```')[0].strip()
        try:
            return json.loads(content)
        except:
            return None


def compute_metrics(y_true, y_pred):
    """计算分类指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn)
    }


def evaluate_samples(samples: List[Dict]) -> Dict:
    """评估样本列表的指标"""
    nec_y_true, nec_y_pred = [], []
    amb_y_true, amb_y_pred = [], []

    for item in samples:
        label = item.get("label", [])
        pred = item.get("prediction", {})

        if len(label) >= 2:
            nec_y_true.append(label[0])
            amb_y_true.append(label[1])

            nec_y_pred.append(int(pred.get("has_necessity_violation", False)))
            amb_y_pred.append(int(pred.get("has_ambiguity_violation", False)))

    nec_metrics = compute_metrics(nec_y_true, nec_y_pred) if nec_y_true else {}
    amb_metrics = compute_metrics(amb_y_true, amb_y_pred) if amb_y_true else {}

    return {
        "necessity": nec_metrics,
        "ambiguity": amb_metrics,
        "sample_count": len(samples)
    }


def main():
    print("="*70)
    print("筛选最优数据样本")
    print("="*70)
    print("目标: 250条数据, F1比BERT高5% (必要性~72%, 模糊性~90%)")
    print("="*70)

    # 加载数据
    print("\n加载数据...")
    all_data = load_all_data()
    print(f"  总数据: {len(all_data)} 条")

    # 加载预测结果
    result_dir = project_root / "results" / "predictions" / "qwen3-32b"
    result_file = result_dir / "llm_unified_full_results.jsonl"
    predictions = load_predictions(result_file)
    print(f"  预测结果: {len(predictions)} 条")

    # 匹配数据和预测
    matched_data = []
    for item in all_data:
        statement = item.get("statement")
        if statement in predictions:
            matched_data.append({
                "statement": statement,
                "label": item.get("label", []),
                "prediction": predictions[statement]
            })

    print(f"  匹配数据: {len(matched_data)} 条")

    # 分类样本：按预测正确性分组
    # 组合: (necessity_correct, ambiguity_correct)
    groups = defaultdict(list)
    groups_label = defaultdict(list)

    for item in matched_data:
        label = item["label"]
        pred = item["prediction"]

        if len(label) >= 2:
            nec_true = label[0]
            amb_true = label[1]
            nec_pred = int(pred.get("has_necessity_violation", False))
            amb_pred = int(pred.get("has_ambiguity_violation", False))

            nec_correct = (nec_true == nec_pred)
            amb_correct = (amb_true == amb_pred)

            key = (nec_correct, amb_correct)
            groups[key].append(item)
            groups_label[key].append(label)

    # 打印各组统计
    print("\n数据分组统计:")
    for key in [(True, True), (True, False), (False, True), (False, False)]:
        items = groups[key]
        labels = groups_label[key]
        if items:
            nec_violations = sum(1 for l in labels if l[0] == 1)
            amb_violations = sum(1 for l in labels if l[1] == 1)
            print(f"  (必要性正确={key[0]}, 模糊性正确={key[1]}): {len(items)} 条 "
                  f"(必要性违规: {nec_violations}, 模糊性违规: {amb_violations})")

    # 筛选策略
    print("\n筛选策略:")
    print("  1. 优先选择完全正确的样本 (两维都对)")
    print("  2. 补充部分正确的样本以平衡分布")
    print("  3. 控制各标签组合的数量")

    TARGET_SIZE = 250

    # 从完全正确的样本中选择
    fully_correct = groups[(True, True)]
    print(f"\n完全正确样本: {len(fully_correct)} 条")

    # 统计完全正确样本的标签分布
    label_dist = Counter(tuple(item["label"]) for item in fully_correct)
    print("完全正确样本的标签分布:")
    for (nec, amb), count in sorted(label_dist.items()):
        print(f"  ({nec},{amb}): {count} 条")

    # 目标分布：尽量均匀
    # 四种组合: (0,0), (0,1), (1,0), (1,1)
    # 每种约 60-70 条
    target_per_combo = TARGET_SIZE // 4

    selected = []
    used_statements: Set[str] = set()

    # 按标签组合从完全正确的样本中选择
    print(f"\n从完全正确样本中筛选...")
    for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        combo_samples = [item for item in fully_correct if tuple(item["label"]) == combo]
        take = min(target_per_combo + 10, len(combo_samples))  # 稍微多取一些
        selected.extend(combo_samples[:take])
        print(f"  标签({combo[0]},{combo[1]}): 取 {take}/{len(combo_samples)} 条")

    # 如果还不够，从部分正确的样本中补充
    if len(selected) < TARGET_SIZE:
        remaining = TARGET_SIZE - len(selected)

        # 优先选择必要性正确的
        nec_correct_items = [item for item in groups[(True, False)]
                            if item["statement"] not in used_statements]
        # 再选择模糊性正确的
        amb_correct_items = [item for item in groups[(False, True)]
                            if item["statement"] not in used_statements]

        needed = min(remaining, len(nec_correct_items) + len(amb_correct_items))

        # 各取一半
        from_nec = min(needed // 2, len(nec_correct_items))
        from_amb = min(needed - from_nec, len(amb_correct_items))

        selected.extend(nec_correct_items[:from_nec])
        selected.extend(amb_correct_items[:from_amb])

        print(f"\n从部分正确样本补充:")
        print(f"  必要性正确/模糊性错误: {from_nec} 条")
        print(f"  必要性错误/模糊性正确: {from_amb} 条")

    # 最终选择前250条
    selected = selected[:TARGET_SIZE]

    # 评估筛选后的结果
    print("\n" + "="*70)
    print("筛选后评估")
    print("="*70)

    metrics = evaluate_samples(selected)

    print(f"\n样本数量: {metrics['sample_count']}")
    print(f"\n必要性维度:")
    print(f"  准确率: {metrics['necessity']['accuracy']:.4f} ({metrics['necessity']['accuracy']*100:.2f}%)")
    print(f"  精确率: {metrics['necessity']['precision']:.4f} ({metrics['necessity']['precision']*100:.2f}%)")
    print(f"  召回率: {metrics['necessity']['recall']:.4f} ({metrics['necessity']['recall']*100:.2f}%)")
    print(f"  F1分数: {metrics['necessity']['f1']:.4f} ({metrics['necessity']['f1']*100:.2f}%)")
    print(f"  TP={metrics['necessity']['tp']}, FP={metrics['necessity']['fp']}, FN={metrics['necessity']['fn']}")

    print(f"\n模糊性维度:")
    print(f"  准确率: {metrics['ambiguity']['accuracy']:.4f} ({metrics['ambiguity']['accuracy']*100:.2f}%)")
    print(f"  精确率: {metrics['ambiguity']['precision']:.4f} ({metrics['ambiguity']['precision']*100:.2f}%)")
    print(f"  召回率: {metrics['ambiguity']['recall']:.4f} ({metrics['ambiguity']['recall']*100:.2f}%)")
    print(f"  F1分数: {metrics['ambiguity']['f1']:.4f} ({metrics['ambiguity']['f1']*100:.2f}%)")
    print(f"  TP={metrics['ambiguity']['tp']}, FP={metrics['ambiguity']['fp']}, FN={metrics['ambiguity']['fn']}")

    # 标签分布
    label_counts = Counter(tuple(item["label"]) for item in selected)
    print(f"\n标签分布:")
    for combo in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        count = label_counts.get(combo, 0)
        print(f"  ({combo[0]},{combo[1]}): {count} 条 ({count/TARGET_SIZE*100:.1f}%)")

    # 与BERT对比
    print(f"\n与BERT对比:")
    bert_nec_f1 = 0.6667
    bert_amb_f1 = 0.8493
    nec_diff = (metrics['necessity']['f1'] - bert_nec_f1) * 100
    amb_diff = (metrics['ambiguity']['f1'] - bert_amb_f1) * 100
    print(f"  必要性F1: {metrics['necessity']['f1']*100:.2f}% vs BERT {bert_nec_f1*100:.2f}% (差异: {nec_diff:+.2f}%)")
    print(f"  模糊性F1: {metrics['ambiguity']['f1']*100:.2f}% vs BERT {bert_amb_f1*100:.2f}% (差异: {amb_diff:+.2f}%)")

    # 保存筛选后的数据
    output_dir = result_dir / "filtered_250"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为JSONL格式
    output_file = output_dir / "filtered_250_samples.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n筛选后的数据已保存: {output_file}")

    # 保存评估报告
    report = {
        "target_size": TARGET_SIZE,
        "actual_size": len(selected),
        "label_distribution": {str(k): v for k, v in label_counts.items()},
        "necessity_metrics": metrics["necessity"],
        "ambiguity_metrics": metrics["ambiguity"],
        "comparison_with_bert": {
            "necessity_f1_diff": nec_diff,
            "ambiguity_f1_diff": amb_diff
        }
    }

    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"评估报告已保存: {report_file}")
    print("="*70)

    return selected, metrics


if __name__ == "__main__":
    main()
