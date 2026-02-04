# LaTeX 论文写作指南

## 文件结构

```
paper/
├── main.tex                  # 主文档
├── chapter3.tex              # 第三章：隐私声明目的合理性分类
├── figures/                  # 图片目录（需创建）
│   ├── pipeline.pdf          # 技术路线图
│   ├── comparison.pdf        # 性能对比图
│   └── ...                   # 其他图片
├── references.bib            # 参考文献（可选）
└── README.md                 # 本文件
```

## 编译方式

### 方法1：使用 XeLaTeX（推荐）

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

### 方法2：使用 LaTeX 编辑器

推荐使用：
- Overleaf（在线）
- TeXShop（macOS）
- TeXStudio（跨平台）
- VS Code + LaTeX Workshop 插件

## 当前状态

✅ 已完成：
- 主文档框架（main.tex）
- 第三章完整结构（chapter3.tex）
- 所有章节标题和小节
- 表格占位符
- 图片占位符

⏳ 待填充：
- 实验数据（表3.2 - 3.6）
- 实验图片（图3.1 - 3.2）
- 具体分析内容
- 案例详细描述

## 数据来源

### 实验数据位置

训练结果保存在：
```
/root/autodl-tmp/roberta_necessity_classifier/training_results/
├── training_history.csv         # 学习曲线数据
├── final_evaluation.json        # 最终评估结果
├── classification_report.txt    # 分类报告
└── confusion_matrix.json        # 混淆矩阵

/root/autodl-tmp/roberta_ambiguity_classifier/training_results/
└── (同上结构)

/root/autodl-tmp/roberta_multilabel_classifier/training_results/
└── (同上结构)
```

### 如何填充数据

1. **从 JSON 文件提取指标**
```python
import json

# 读取评估结果
with open('final_evaluation.json', 'r') as f:
    data = json.load(f)

# 获取指标
accuracy = data['final_metrics']['accuracy']
precision = data['final_metrics']['precision']
recall = data['final_metrics']['recall']
f1 = data['final_metrics']['f1']
```

2. **从 CSV 文件绘制学习曲线**
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练历史
df = pd.read_csv('training_history.csv')

# 绘制 F1 曲线
plt.plot(df['epoch'], df['eval_f1'])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.savefig('learning_curve.pdf')
```

## 表格填写说明

### 表3.1：数据集标签分布
已填写（基于1137条样本的实际分布）

### 表3.2：BERT基线模型性能
等待训练完成后，从 `final_evaluation.json` 填入

### 表3.3：LLM分类器性能
等待 `evaluate_classification.py` 运行完成后填入

### 表3.4：一致性分析
等待 `run_all_classifications.py` 运行完成后填入

### 表3.5：提示策略消融实验
需要额外设计实验（可选择性添加）

## 图片生成建议

### 图3.1：技术路线图
建议工具：draw.io, PowerPoint, Adobe Illustrator
内容：三阶段流程图（提示工程 → SFT → 蒸馏）

### 图3.2：性能对比图
建议工具：Python (matplotlib), Excel, Origin
数据来源：各模型的 `final_evaluation.json`

示例代码：
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['多标签\nBERT', '必要性\nBERT', '表述模糊\nBERT',
          '统一\nLLM', '必要性\nLLM', '表述模糊\nLLM']
f1_scores = [0.75, 0.80, 0.78, 0.85, 0.88, 0.86]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color=['skyblue']*3 + ['lightcoral']*3)
plt.ylabel('F1 Score')
plt.title('模型性能对比')
plt.ylim([0.7, 0.9])
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/comparison.pdf', format='pdf')
```

## 写作建议

### 1. 结构完整性
第三章已包含所有必要部分：
- 引言
- 问题定义
- 方法（三阶段）
- 实验（数据、设置、结果、分析）
- 讨论
- 小结

### 2. 实验描述重点
- 清晰说明数据集来源和标注过程
- 详细描述实验设置（超参数、优化器等）
- 客观呈现实验结果
- 深入分析对比结果

### 3. 图表配合
- 每个图表都应在正文中引用
- 图表应有清晰标题和说明
- 数据应准确无误

### 4. 案例分析
- 选择代表性案例（正确/错误）
- 分析模型判别依据
- 说明方法优势和局限

## 快速开始

1. **编译主文档**
```bash
xelatex main.tex
```

2. **查看第三章预览**
```bash
xelatex chapter3.tex
```

3. **开始填充数据**
根据实验结果逐步填写表格和图表

## 后续工作

当前代码可支撑：
- ✅ 3.2节：BERT基线实验
- ✅ 3.3节：LLM提示工程实验
- ✅ 3.4节：对比分析
- ⏳ 3.5节：消融实验（需要额外设计）
- ⏳ 3.6节：案例分析（需要整理典型案例）

## 代码与论文对应

| 论文内容 | 对应代码/数据 |
|---------|-------------|
| 表3.2 BERT性能 | `train_*.py` → `final_evaluation.json` |
| 表3.3 LLM性能 | `classify_*.py` → `evaluate_classification.py` |
| 表3.4 一致性 | `run_all_classifications.py` → `classifier_comparison.json` |
| 图3.2 性能对比 | 综合上述数据 |
| 案例分析 | `results/predictions/*.jsonl` |
