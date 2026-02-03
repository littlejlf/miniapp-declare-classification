# BERT 单任务分类模型训练指南

本文档介绍如何使用独立的 BERT 模型分别训练必要性违规分类和表述模糊违规分类任务。

---

## 一、模型架构

### 方案对比

| 方案 | 模型数量 | 特点 | 适用场景 |
|------|---------|------|---------|
| 多标签分类 | 1个模型 | 同时输出两个维度 | 两个任务相关性高，资源受限 |
| **单任务分类** | **2个独立模型** | **每个模型专注一个任务** | **任务独立，追求最佳性能** |

### 本项目方案

本项目采用**单任务分类**方案：

1. **必要性分类模型** (`roberta_necessity_classifier`)
   - 输入: 隐私声明文本
   - 输出: 正常(0) / 违规(1)
   - 判断: 目的异常、可替代冗余

2. **表述模糊分类模型** (`roberta_ambiguity_classifier`)
   - 输入: 隐私声明文本
   - 输出: 清晰(0) / 模糊(1)
   - 判断: 语义重复、场景缺失、表述宽泛

---

## 二、类别权重处理

### 为什么需要类别权重？

当数据集中正负样本比例不平衡时，模型会倾向于预测多数类。类别权重通过给少数类更高的损失权重来平衡训练。

### 权重计算方法

#### 方法1: Balanced（默认）

```python
weight = n_samples / (n_classes * n_samples_per_class)
```

示例：
- 正常样本: 900个 → 权重 = 1137 / (2 * 900) = 0.63
- 违规样本: 237个 → 权重 = 1137 / (2 * 237) = 2.40

#### 方法2: 手动指定

在脚本中修改 `MANUAL_WEIGHTS`:

```python
# [负类权重, 正类权重]
MANUAL_WEIGHTS = [1.0, 2.0]  # 正类权重是负类的2倍
```

---

## 三、使用方式

### 3.1 单独训练必要性模型

```bash
python experiments/baseline/train_necessity.py
```

**输出目录**: `results/models/roberta_necessity_classifier/`

**输出文件**:
- `pytorch_model.bin` - 模型权重
- `config.json` - 模型配置
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表
- `config.json` - 训练配置和评估结果

### 3.2 单独训练表述模糊模型

```bash
python experiments/baseline/train_ambiguity.py
```

**输出目录**: `results/models/roberta_ambiguity_classifier/`

### 3.3 批量训练两个模型

```bash
python experiments/baseline/train_both_tasks.py
```

---

## 四、训练参数配置

### 主要参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `MODEL_NAME` | `hfl/chinese-roberta-wwm-ext` | 预训练模型 |
| `NUM_EPOCHS` | 5 | 训练轮数 |
| `BATCH_SIZE` | 16 | 批次大小 |
| `LEARNING_RATE` | 2e-5 | 学习率 |
| `WARMUP_STEPS` | 100 | 预热步数 |
| `WEIGHT_DECAY` | 0.01 | 权重衰减 |
| `MANUAL_WEIGHTS` | None | 手动指定权重 |

### 修改参数

编辑训练脚本中的配置参数：

```python
# train_necessity.py 或 train_ambiguity.py

# 训练参数
NUM_EPOCHS = 10        # 增加训练轮数
BATCH_SIZE = 32        # 增加批次大小（需要更多显存）
LEARNING_RATE = 1e-5   # 降低学习率

# 类别权重
MANUAL_WEIGHTS = [1.0, 3.0]  # 手动设置正类权重为负类的3倍
```

---

## 五、训练输出说明

### 训练日志示例

```
类别分布统计:
  类别 0: 910 样本 (80.04%)
  类别 1: 227 样本 (19.96%)
计算得到的类别权重: [0.6247211582421299, 2.5024994251642714]

训练集: 910 条
验证集: 228 条
```

### 评估指标

```
最终评估结果
============================================================
  eval_loss: 0.3245
  eval_accuracy: 0.8947
  eval_precision: 0.8571
  eval_recall: 0.7500
  eval_f1: 0.8000
```

### 分类报告

```
              precision    recall  f1-score   support

          正常     0.9032    0.9538    0.9278       130
          违规     0.8571    0.7500    0.8000        98

    accuracy                         0.8947       228
   macro avg     0.8802    0.8519    0.8639       228
weighted avg     0.8935    0.8947    0.8927       228

混淆矩阵:
  TN=124, FP=6
  FN=18, TP=80
```

---

## 六、推理使用

### 加载模型进行预测

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载必要性模型
necessity_tokenizer = AutoTokenizer.from_pretrained("results/models/roberta_necessity_classifier")
necessity_model = AutoModelForSequenceClassification.from_pretrained("results/models/roberta_necessity_classifier")

# 加载表述模糊模型
ambiguity_tokenizer = AutoTokenizer.from_pretrained("results/models/roberta_ambiguity_classifier")
ambiguity_model = AutoModelForSequenceClassification.from_pretrained("results/models/roberta_ambiguity_classifier")

def predict(text):
    # 预测必要性
    nec_inputs = necessity_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    nec_outputs = necessity_model(**nec_inputs)
    nec_pred = torch.argmax(nec_outputs.logits, dim=1).item()

    # 预测表述模糊
    amb_inputs = ambiguity_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    amb_outputs = ambiguity_model(**amb_inputs)
    amb_pred = torch.argmax(amb_outputs.logits, dim=1).item()

    return {
        "has_necessity_violation": bool(nec_pred),
        "has_ambiguity_violation": bool(amb_pred)
    }

# 使用示例
text = "为了系统开发，开发者收集你的位置信息"
result = predict(text)
print(result)
# 输出: {'has_necessity_violation': True, 'has_ambiguity_violation': False}
```

---

## 七、常见问题

### Q1: 显存不足怎么办？

A: 减小 `BATCH_SIZE` 或使用梯度累积：

```python
# 减小批次大小
BATCH_SIZE = 8

# 或使用梯度累积（在 TrainingArguments 中添加）
gradient_accumulation_steps=2
```

### Q2: 如何调整权重比例？

A: 根据验证集表现调整：

```python
# 如果正类召回率低，增加正类权重
MANUAL_WEIGHTS = [1.0, 3.0]

# 如果负类准确率低，降低正类权重
MANUAL_WEIGHTS = [1.0, 1.5]
```

### Q3: 两个模型应该同时训练还是分开训练？

A: 建议分开训练，原因：
- 可以针对每个任务调整超参数
- 便于调试和比较
- 可以使用不同的类别权重

### Q4: 训练多久合适？

A: 建议：
- 初始训练: 3-5 个 epoch
- 如果验证集指标仍在提升: 增加到 10 个 epoch
- 使用 `load_best_model_at_end=True` 自动保存最佳模型

---

## 八、输出目录结构

```
results/models/
├── roberta_necessity_classifier/
│   ├── pytorch_model.bin           # 模型权重
│   ├── config.json                 # Transformer配置
│   ├── tokenizer_config.json       # 分词器配置
│   ├── vocab.txt                   # 词汇表
│   ├── special_tokens_map.json     # 特殊token
│   ├── config.json                 # 训练配置（包含类别权重）
│   └── checkpoint-*/               # 训练检查点
└── roberta_ambiguity_classifier/
    └── (同上结构)
```

---

## 九、与多标签模型对比

| 指标 | 多标签模型 | 单任务模型 |
|------|-----------|-----------|
| 模型数量 | 1个 | 2个 |
| 训练时间 | 较短 | 稍长 |
| 推理速度 | 更快 | 需要两次推理 |
| 灵活性 | 低 | 高（可独立优化） |
| 性能 | 一般 | **通常更好** |
| 类别权重 | 难以分别设置 | **可独立设置** |

---

## 十、参考链接

- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/)
- [中文 RoBERTa 模型](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
- [分类任务不平衡处理](https://scikit-learn.org/stable/auto_examples/model_selection/plot_class_weight.html)
