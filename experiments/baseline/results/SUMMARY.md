# Baseline 实验结果汇总

## 实验概述

本次实验训练了三个基于中文 RoBERTa 的模型：
1. **必要性分类模型** - 判断数据收集是否必要（二分类）
2. **清晰性分类模型** - 判断声明表述是否清晰（二分类）
3. **多标签分类模型** - 同时预测必要性和清晰性（多标签）

## 数据集信息

- 总样本数：1,137 条
- 训练集：909 条 (80%)
- 验证集：228 条 (20%)
- 使用分层采样保持标签分布

## 模型配置

- 基础模型：`hfl/chinese-roberta-wwm-ext`
- 最大序列长度：128
- 批次大小：16
- 学习率：2e-5
- 训练轮数：10 epochs（带早停）
- 优化器：AdamW
- 学习率调度：余弦退火
- 类别权重：自动平衡计算（单任务模型）

---

## 1. 必要性分类模型结果（单任务）

### 最终指标
| 指标 | 值 |
|-----|-----|
| **Accuracy** | 79.82% |
| **Precision** | 64.79% |
| **Recall** | 68.66% |
| **F1-Score** | 66.67% |
| Loss | 0.9894 |

### 详细分类报告

| 类别 | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| 正常 | 86.62% | 84.47% | 85.53% | 161 |
| 违规 | 64.79% | 68.66% | 66.67% | 67 |

### 混淆矩阵

```
                预测正常   预测违规
实际正常 (TN)      136        25
实际违规 (FN)       21        46
```

### 类别分布
- 训练集：正常 641 (70.52%) / 违规 268 (29.48%)
- 类别权重：[0.709, 1.696]

---

## 2. 清晰性分类模型结果（单任务）

### 最终指标
| 指标 | 值 |
|-----|-----|
| **Accuracy** | 85.53% |
| **Precision** | 82.30% |
| **Recall** | 87.74% |
| **F1-Score** | 84.93% |
| Loss | 0.5063 |

### 详细分类报告

| 类别 | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| 清晰 | 88.70% | 83.61% | 86.08% | 122 |
| 模糊 | 82.30% | 87.74% | 84.93% | 106 |

### 混淆矩阵

```
                预测清晰   预测模糊
实际清晰 (TN)      102        20
实际模糊 (FN)       13        93
```

### 类别分布
- 训练集：清晰 486 (53.47%) / 模糊 423 (46.53%)
- 类别权重：[0.935, 1.074]

---

## 3. 多标签分类模型结果（联合任务）

### 最终指标
| 指标 | 值 |
|-----|-----|
| **F1-Macro** | 78.77% |
| **Precision-Macro** | 80.20% |
| **Recall-Macro** | 77.65% |
| **Subset Accuracy** | 73.25% |
| Loss | 0.4549 |

### 指标说明

- **F1-Macro**: 两个标签的F1分数的宏平均
- **Precision-Macro**: 两个标签的精确率宏平均
- **Recall-Macro**: 两个标签的召回率宏平均
- **Subset Accuracy**: 要求所有标签都预测正确才算准确

---

## 对比分析

| 模型 | Accuracy | Precision | Recall | F1-Score |
|-----|----------|-----------|--------|----------|
| 必要性分类（单任务） | 79.82% | 64.79% | 68.66% | 66.67% |
| 清晰性分类（单任务） | 85.53% | 82.30% | 87.74% | 84.93% |
| 多标签分类（联合） | 73.25%* | 80.20% | 77.65% | 78.77% |

*注：多标签模型的 Accuracy 使用的是 Subset Accuracy（要求所有标签预测正确）

### 结论

1. **单任务模型表现更优** - 分别训练两个模型在各自任务上表现更好
2. **清晰性分类最容易** - 无论是单任务还是多任务，清晰性判断都更准确
3. **必要性分类更具挑战** - "必要性"判断需要更多上下文和业务理解
4. **多标签模型的优势** - 一次推理预测两个维度，推理速度更快

---

## 文件结构

```
results/
├── roberta_necessity_classifier/          # 必要性分类模型
│   ├── model.safetensors                   # 模型权重 (409MB)
│   ├── config.json                         # 模型配置
│   ├── tokenizer.json                      # 分词器
│   ├── vocab.txt                           # 词典
│   └── training_results/                   # 训练结果
│       ├── training_history.csv
│       ├── final_evaluation.json
│       ├── classification_report.txt
│       └── confusion_matrix.json
│
├── roberta_ambiguity_classifier/           # 清晰性分类模型
│   ├── model.safetensors                   # 模型权重 (409MB)
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── training_results/
│       ├── training_history.csv
│       ├── final_evaluation.json
│       ├── classification_report.txt
│       └── confusion_matrix.json
│
└── roberta_multilabel_classifier/          # 多标签分类模型
    ├── model.safetensors                   # 模型权重 (409MB)
    ├── config.json
    ├── tokenizer.json
    ├── vocab.txt
    └── training_results/
        ├── training_history.csv
        ├── final_evaluation.json
        └── evaluation_report.txt
```

**注意**: 所有模型实际存储在 `/root/autodl-tmp/`，项目目录中的是软链接。

---

## 使用方法

### 加载单任务模型

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载必要性分类模型
necessity_tokenizer = AutoTokenizer.from_pretrained("results/roberta_necessity_classifier")
necessity_model = AutoModelForSequenceClassification.from_pretrained("results/roberta_necessity_classifier")

# 加载清晰性分类模型
ambiguity_tokenizer = AutoTokenizer.from_pretrained("results/roberta_ambiguity_classifier")
ambiguity_model = AutoModelForSequenceClassification.from_pretrained("results/roberta_ambiguity_classifier")

# 预测
text = "收集用户设备信息用于优化应用体验"
inputs = necessity_tokenizer(text, return_tensors="pt")
outputs = necessity_model(**inputs)
prediction = outputs.logits.argmax().item()  # 0=正常, 1=违规
```

### 加载多标签模型

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载多标签分类模型
tokenizer = AutoTokenizer.from_pretrained("results/roberta_multilabel_classifier")
model = AutoModelForSequenceClassification.from_pretrained("results/roberta_multilabel_classifier")

# 预测（同时输出两个标签）
text = "收集用户设备信息用于优化应用体验"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# 使用sigmoid将logits转换为概率，然后用阈值0.5转换为0/1
probs = torch.sigmoid(outputs.logits).squeeze()
preds = (probs > 0.5).int()

print(f"必要性违规: {preds[0].item()} (0=正常, 1=违规)")
print(f"清晰性违规: {preds[1].item()} (0=清晰, 1=模糊)")
```

---

## 训练时间

| 模型 | 训练时间 | Epochs |
|-----|----------|--------|
| 必要性分类 | ~30 秒 | 6 epochs (早停) |
| 清晰性分类 | ~30 秒 | 4 epochs (早停) |
| 多标签分类 | ~41 秒 | 10 epochs (完整) |
| **总计** | **~1.7 分钟** | - |

---

## 训练环境

- 平台：AutoDL GPU 实例
- 数据存储：`/root/autodl-tmp/` (46G 可用，高速临时存储)
- 模型访问：项目目录软链接指向数据存储位置
- Python：3.12
- 框架：PyTorch + Transformers

---

*报告生成时间：2026-02-27*
