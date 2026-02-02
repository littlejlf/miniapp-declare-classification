# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import numpy as np
import json  # 用于JSONL数据
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# 导入配置
from utils.logger import get_logger, configure_logging
from utils.metrics import format_metrics_report

# --- 1. 配置与参数 ---
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# 使用相对路径，适配AutoDL和本地环境
DATA_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
OUTPUT_DIR = project_root / "results" / "models" / "roberta_privacy_multilabel_classifier"
NUM_EPOCHS = 3
BATCH_SIZE = 16

# 配置日志
logger = get_logger(__name__)

# --- 2. 准备标签映射 ---
# 【修改】 - 定义多标签的名称
label_names = ["has_necessity_violation", "has_ambiguity_violation"]
NUM_LABELS = len(label_names)

# --- 3. 加载和预处理数据 ---
logger.info(f"加载数据从: {DATA_FILE}")

# 从JSONL文件加载数据
data = []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line.strip())
            # 提取文本和标签
            # 假设JSONL格式: {"appid": "...", "declare": "...", "necessity": 0/1, "clarity": 0/1}
            data.append({
                'text': item.get('declare', ''),
                'labels': [
                    item.get('necessity', 0),  # 必要性违规
                    item.get('clarity', 0)      # 清晰性违规
                ]
            })

df = pd.DataFrame(data)
logger.info(f"加载数据完成，共 {len(df)} 条")

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
logger.info(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")

# 转换为Hugging Face的Dataset格式
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})
logger.info(f"数据示例: {dataset_dict['train'][0]}")

# --- 4. 初始化Tokenizer和模型 ---
logger.info("加载Tokenizer和模型用于多标签分类...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 【核心修改】 - 告诉模型这是一个多标签分类问题
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"  # <--- 关键参数！
)

# --- 5. 定义数据处理函数 ---
def tokenize_function(examples):
    #todo 这里要改 加理由
    return tokenizer(examples["text"], truncation=True, max_length=128)

logger.info("对数据集进行Tokenization...")
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# 移除不再需要的列并设置格式
# 注意: 'labels' 列现在是核心部分，不能移除
columns_to_remove = [col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]]
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 6. 定义多标签评估指标 ---
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    
    # Sigmoid输出是logits，需要转换成概率，然后根据阈值（0.5）转换为0或1
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    
    # 计算宏平均F1分数，这在多标签中是很好的指标
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    # 计算子集准确率（要求所有标签都预测正确）
    subset_acc = accuracy_score(labels, preds)
    
    return {'f1_macro': f1, 'precision_macro': precision, 'recall_macro': recall, 'subset_accuracy': subset_acc}

# --- 7. 定义训练参数 ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro", # 使用新的F1指标名
    greater_is_better=True,
    weight_decay=0.01,
    warmup_steps=500
)

# --- 8. 初始化并开始训练 ---
# 【修改】 - 使用标准的Trainer，不再需要CustomTrainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("开始多标签分类训练...")
trainer.train()

# --- 9. 评估并保存模型 ---
logger.info("训练完成，评估最佳模型...")
eval_results = trainer.evaluate()
logger.info(f"最终验证集评估结果: {eval_results}")

logger.info(f"将最佳模型保存到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
logger.info("模型保存完成")