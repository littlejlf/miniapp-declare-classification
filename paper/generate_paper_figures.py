#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文图表生成脚本

从训练结果自动生成第三章所需的图表和数据
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class PaperFigureGenerator:
    """论文图表生成器"""

    def __init__(self, results_base_dir="/root/autodl-tmp"):
        self.results_base_dir = Path(results_base_dir)

    def load_evaluation_results(self, model_name):
        """
        加载模型的评估结果

        Args:
            model_name: 模型目录名

        Returns:
            评估结果字典
        """
        result_file = self.results_base_dir / model_name / "training_results" / "final_evaluation.json"

        if not result_file.exists():
            print(f"警告: {result_file} 不存在")
            return None

        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_table_3_1(self):
        """生成表3.1：数据集标签分布"""
        print("\n表3.1：数据集标签分布")
        print("-" * 60)
        print("| 维度 | 负样本 | 正样本 | 总计 |")
        print("|------|--------|--------|------|")
        print("| 必要性违规 | 910 (80.0%) | 227 (20.0%) | 1137 |")
        print("| 表述模糊违规 | 608 (53.5%) | 529 (46.5%) | 1137 |")
        print("-" * 60)

    def generate_table_3_2(self):
        """生成表3.2：BERT基线模型性能"""
        print("\n表3.2：BERT基线模型性能")
        print("-" * 70)

        models = [
            ("roberta_multilabel_classifier", "多标签分类"),
            ("roberta_necessity_classifier", "必要性（单任务）"),
            ("roberta_ambiguity_classifier", "表述模糊（单任务）")
        ]

        print("| 模型 | Accuracy | Precision | Recall | F1 |")
        print("|------|----------|-----------|--------|-----|")

        for model_dir, model_name in models:
            results = self.load_evaluation_results(model_dir)
            if results:
                m = results['final_metrics']
                print(f"| {model_name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                      f"{m['recall']:.4f} | {m['f1']:.4f} |")

        print("-" * 70)

        return models  # 返回用于后续绘图

    def generate_table_3_3(self):
        """生成表3.3：LLM分类器性能"""
        print("\n表3.3：LLM分类器性能")
        print("-" * 70)
        print("| 分类器 | Accuracy | Precision | Recall | F1 |")
        print("|--------|----------|-----------|--------|-----|")
        print("| 统一分类 | 待运行 | 待运行 | 待运行 | 待运行 |")
        print("| 必要性（独立） | 待运行 | 待运行 | 待运行 | 待运行 |")
        print("| 表述模糊（独立） | 待运行 | 待运行 | 待运行 | 待运行 |")
        print("-" * 70)

    def generate_table_3_4(self):
        """生成表3.4：一致性分析"""
        # 从 run_all_classifications.py 的输出中读取
        comparison_file = self.results_base_dir.parent / "results/predictions/classifier_comparison.json"

        if comparison_file.exists():
            with open(comparison_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("\n表3.4：分类器一致性分析")
            print("-" * 50)
            print("| 维度 | 一致率 | Cohen's Kappa |")
            print("|------|--------|----------------|")

            for dim in ['necessity', 'ambiguity']:
                consistency = data[dim]['consistency']
                print(f"| {dim.capitalize()} | {consistency['agreement_rate']:.4f} | "
                      f"{consistency['cohen_kappa']:.4f} |")

            print("-" * 50)
        else:
            print("\n表3.4：一致性分析（待运行 run_all_classifications.py）")

    def plot_comparison_figure(self, bert_models, output_path="figures/comparison.pdf"):
        """
        绘制图3.2：BERT vs LLM 性能对比

        Args:
            bert_models: BERT模型列表
            output_path: 输出路径
        """
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 收集BERT数据
        bert_data = []
        for model_dir, model_name in bert_models:
            results = self.load_evaluation_results(model_dir)
            if results:
                bert_data.append({
                    'name': model_name,
                    'f1': results['final_metrics']['f1'],
                    'type': 'BERT'
                })

        # LLM数据（待填充）
        llm_data = [
            {'name': '统一分类', 'f1': 0, 'type': 'LLM'},
            {'name': '必要性（独立）', 'f1': 0, 'type': 'LLM'},
            {'name': '表述模糊（独立）', 'f1': 0, 'type': 'LLM'}
        ]

        # 合并数据
        all_data = bert_data + llm_data

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(all_data))
        width = 0.6

        colors = ['#3498db'] * len(bert_data) + ['#e74c3c'] * len(llm_data)
        bars = ax.bar(x, [d['f1'] for d in all_data], width,
                      color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for i, (bar, data) in enumerate(zip(bars, all_data)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('BERT vs LLM 性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d['name'] for d in all_data], rotation=15, ha='right')
        ax.set_ylim([0, 1.0])
        ax.axvline(x=len(bert_data) - 0.5, color='gray', linestyle='--', linewidth=2)
        ax.grid(axis='y', alpha=0.3)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='BERT'),
            Patch(facecolor='#e74c3c', label='LLM')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"\n✓ 对比图已保存: {output_path}")

        plt.close()

    def plot_learning_curves(self, output_dir="figures"):
        """绘制学习曲线"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        models = [
            ("roberta_necessity_classifier", "必要性"),
            ("roberta_ambiguity_classifier", "表述模糊")
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, (model_dir, model_name) in enumerate(models):
            history_file = self.results_base_dir / model_dir / "training_results" / "training_history.csv"

            if history_file.exists():
                df = pd.read_csv(history_file)

                ax = axes[idx]

                # 绘制训练曲线
                ax.plot(df['epoch'], df['eval_f1'], 'o-', label='F1', linewidth=2)
                ax.plot(df['epoch'], df['eval_precision'], 's--', label='Precision', linewidth=1.5)
                ax.plot(df['epoch'], df['eval_recall'], '^.--', label='Recall', linewidth=1.5)

                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Score', fontsize=11)
                ax.set_title(f'{model_name}分类 - 学习曲线', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.0])

        plt.tight_layout()
        output_path = output_dir / "learning_curves.pdf"
        plt.savefig(output_path, format='pdf', dpi=300)
        print(f"✓ 学习曲线已保存: {output_path}")

        plt.close()

    def generate_latex_tables(self):
        """生成LaTeX表格代码"""

        # 表3.2
        latex_table_3_2 = r"""
\begin{table}[htbp]
    \centering
    \caption{BERT基线模型性能}
    \label{tab:baseline_results}
    \begin{tabular}{lcccc}
        \toprule
        模型 & Accuracy & Precision & Recall & F1 \\
        \midrule
"""

        models = [
            ("roberta_multilabel_classifier", "多标签分类"),
            ("roberta_necessity_classifier", "必要性（单任务）"),
            ("roberta_ambiguity_classifier", "表述模糊（单任务）")
        ]

        for model_dir, model_name in models:
            results = self.load_evaluation_results(model_dir)
            if results:
                m = results['final_metrics']
                latex_table_3_2 += f"        {model_name} & {m['accuracy']:.4f} & {m['precision']:.4f} & "
                latex_table_3_2 += f"{m['recall']:.4f} & {m['f1']:.4f} \\\\\n"

        latex_table_3_2 += r"        \bottomrule" + "\n"
        latex_table_3_2 += r"    \end{tabular}" + "\n"
        latex_table_3_2 += r"\end{table}"

        print("\nLaTeX代码 - 表3.2：")
        print(latex_table_3_2)

        # 保存到文件
        with open('table_3_2.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table_3_2)
        print("✓ LaTeX表格代码已保存: table_3_2.tex")

    def generate_all(self):
        """生成所有图表和数据"""
        print("=" * 70)
        print("论文第三章图表生成")
        print("=" * 70)

        # 生成表格数据
        self.generate_table_3_1()
        bert_models = self.generate_table_3_2()
        self.generate_table_3_3()
        self.generate_table_3_4()

        # 生成图表
        print("\n生成图表...")
        self.plot_comparison_figure(bert_models)
        self.plot_learning_curves()

        # 生成LaTeX代码
        print("\n生成LaTeX表格代码...")
        self.generate_latex_tables()

        print("\n" + "=" * 70)
        print("✓ 所有图表和数据生成完成！")
        print("=" * 70)


if __name__ == "__main__":
    import sys

    # 支持命令行参数指定结果目录
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp"

    generator = PaperFigureGenerator(results_dir)
    generator.generate_all()
