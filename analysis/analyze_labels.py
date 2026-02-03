# -*- coding: utf-8 -*-
"""
生成数据标签分布的可视化图表
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import Counter

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    data_file = Path('data/raw/aggregate_datas_label.jsonl')

    # 读取数据
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                labels = item.get('label', [0, 0])
                data.append({
                    'necessity': labels[0],
                    'clarity': labels[1] if len(labels) > 1 else 0
                })

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('小程序隐私声明数据集 - 标签分布分析', fontsize=16, fontweight='bold')

    # 1. 标签组合分布（饼图）
    label_combinations = Counter((d['necessity'], d['clarity']) for d in data)
    labels = ['无违规 (0,0)', '仅清晰性违规 (0,1)', '仅必要性违规 (1,0)', '双重违规 (1,1)']
    sizes = [label_combinations[(0,0)], label_combinations[(0,1)],
             label_combinations[(1,0)], label_combinations[(1,1)]]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    explode = (0.05, 0.05, 0.05, 0.05)

    axes[0, 0].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
    axes[0, 0].set_title('标签组合分布', fontsize=14, fontweight='bold')

    # 2. 标签组合分布（柱状图）
    categories = ['无违规\n(0,0)', '仅清晰性\n(0,1)', '仅必要性\n(1,0)', '双重违规\n(1,1)']
    counts = [label_combinations[(0,0)], label_combinations[(0,1)],
              label_combinations[(1,0)], label_combinations[(1,1)]]

    bars = axes[0, 1].bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('各类别样本数量', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('样本数量', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 在柱子上添加数值
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11)

    # 3. 单独维度统计
    necessity_violations = sum(1 for d in data if d['necessity'] == 1)
    clarity_violations = sum(1 for d in data if d['clarity'] == 1)
    no_violations = sum(1 for d in data if d['necessity'] == 0 and d['clarity'] == 0)

    dim_labels = ['无违规', '必要性违规', '清晰性违规']
    dim_counts = [no_violations, necessity_violations, clarity_violations]
    dim_colors = ['#66c2a5', '#fc8d62', '#8da0cb']

    bars2 = axes[1, 0].bar(dim_labels, dim_counts, color=dim_colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('单独维度统计', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('样本数量', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11)

    # 4. 数据集统计摘要
    axes[1, 1].axis('off')
    summary_text = f"""
    数据集统计摘要
    ═══════════════════

    总样本数: {len(data)} 条

    有违规样本: {sum(1 for d in data if d['necessity'] == 1 or d['clarity'] == 1)} 条
    无违规样本: {no_violations} 条

    ───────────────

    必要性违规率: {necessity_violations/len(data)*100:.2f}%
    清晰性违规率: {clarity_violations/len(data)*100:.2f}%

    ───────────────

    不平衡比例: {max(label_combinations.values())/min(label_combinations.values()):.2f}:1
    平衡状态: 轻微不平衡

    ───────────────

    标签分布:
    • (0,0): {label_combinations[(0,0)]} 条 ({label_combinations[(0,0)]/len(data)*100:.1f}%)
    • (0,1): {label_combinations[(0,1)]} 条 ({label_combinations[(0,1)]/len(data)*100:.1f}%)
    • (1,0): {label_combinations[(1,0)]} 条 ({label_combinations[(1,0)]/len(data)*100:.1f}%)
    • (1,1): {label_combinations[(1,1)]} 条 ({label_combinations[(1,1)]/len(data)*100:.1f}%)
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # 保存图表
    output_path = Path('analysis/label_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")

    plt.show()

if __name__ == "__main__":
    create_visualizations()
