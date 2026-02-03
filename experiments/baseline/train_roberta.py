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
    EarlyStoppingCallback,
)

# 导入配置
from utils.logger import get_logger, configure_logging
from utils.metrics import format_metrics_report

# --- 1. 配置与参数 ---
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# 使用相对路径，适配AutoDL和本地环境
DATA_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
# AutoDL 临时存储目录（训练速度更快）
OUTPUT_DIR = Path("/root/autodl-tmp/roberta_multilabel_classifier")

# 训练参数（优化后的配置）
NUM_EPOCHS = 10               # 增加上限，配合早停
BATCH_SIZE = 16               # 保持不变
LEARNING_RATE = 2e-5          # BERT微调标准值
WARMUP_STEPS = 40             # 减少到约0.7个epoch
WEIGHT_DECAY = 0.01           # 防止过拟合
MAX_GRAD_NORM = 1.0           # 梯度裁剪

# 早停配置
EARLY_STOPPING_PATIENCE = 3   # 3个epoch无改善就停止

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
            # JSONL格式: {"statement": "...", "label": [necessity, clarity]}
            labels = item.get('label', [0, 0])
            data.append({
                'text': item.get('statement', ''),
                'labels': labels  # [necessity, clarity]
            })

df = pd.DataFrame(data)
logger.info(f"加载数据完成，共 {len(df)} 条")

# 确保labels是list类型（多标签格式）
df['labels'] = df['labels'].apply(lambda x: x if isinstance(x, list) else [x])

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
    # Tokenize 文本
    tokenized = tokenizer(examples["text"], truncation=True, max_length=128)
    return tokenized

logger.info("对数据集进行Tokenization...")
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# 移除不再需要的列（保留labels）
# 注意: 'labels' 列现在是核心部分，不能移除
columns_to_remove = [col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]]
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)

# 【关键修复】对于多标签分类，labels必须是float类型
# 方法1: 不使用set_format，保持numpy数组
# 方法2: 使用自定义的DataCollator来转换类型

class MultiLabelDataCollator(DataCollatorWithPadding):
    """自定义DataCollator，确保labels是float类型"""
    def __call__(self, features):
        batch = super().__call__(features)
        # 将labels从LongTensor转换为FloatTensor
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
        return batch

data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

# 设置格式
tokenized_datasets.set_format("torch")

# 验证labels类型
sample_label = tokenized_datasets["train"][0]["labels"]
logger.info(f"labels类型（转换前）: {type(sample_label)}, 内容: {sample_label}")

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
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=MAX_GRAD_NORM,      # 梯度裁剪
    lr_scheduler_type="cosine",        # 余弦退火学习率调度
    save_total_limit=1,                # 只保留最佳模型
    report_to=None,                    # 不使用wandb
    seed=42,                           # 随机种子，确保可复现
)

# --- 8. 初始化并开始训练 ---
# 定义带历史记录的Trainer
class HistoryTrainer(Trainer):
    """带训练历史记录的Trainer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = []

    def log(self, logs: dict) -> None:
        super().log(logs)
        if any(key.startswith('eval_') for key in logs.keys()):
            self.training_history.append(logs.copy())

# 早停回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=0.001
)

trainer = HistoryTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]  # 添加早停回调
)

logger.info("开始多标签分类训练...")
trainer.train()

# --- 9. 评估并保存模型 ---
logger.info("训练完成，评估最佳模型...")
eval_results = trainer.evaluate()
logger.info(f"最终验证集评估结果: {eval_results}")

logger.info(f"将最佳模型保存到 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)

# --- 10. 保存训练结果（用于论文画图）---
results_dir = OUTPUT_DIR / "training_results"
results_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"\n保存训练结果到: {results_dir}")

import csv
from datetime import datetime

# 10.1 保存训练历史CSV
if trainer.training_history:
    history_file = results_dir / "training_history.csv"
    with open(history_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = sorted({k for log in trainer.training_history for k in log.keys()
                        if not k.startswith('_')})
        writer.writerow(headers)
        for log in trainer.training_history:
            writer.writerow([log.get(h, '') for h in headers])
    logger.info(f"  训练历史已保存: {history_file}")

# 10.2 保存最终评估结果JSON
final_results = {
    'task': 'multilabel_classification',
    'model_name': MODEL_NAME,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_epochs': NUM_EPOCHS,
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'final_metrics': {
        'f1_macro': float(eval_results.get('eval_f1_macro', 0)),
        'precision_macro': float(eval_results.get('eval_precision_macro', 0)),
        'recall_macro': float(eval_results.get('eval_recall_macro', 0)),
        'subset_accuracy': float(eval_results.get('eval_subset_accuracy', 0)),
        'loss': float(eval_results.get('eval_loss', 0))
    }
}

final_results_file = results_dir / "final_evaluation.json"
with open(final_results_file, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
logger.info(f"  最终评估结果已保存: {final_results_file}")

# 10.3 保存评估报告TXT
report_file = results_dir / "evaluation_report.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f"多标签分类模型 - 评估报告\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"模型: {MODEL_NAME}\n")
    f.write(f"任务: 多标签分类（必要性 + 表述模糊）\n")
    f.write(f"训练轮数: {NUM_EPOCHS}\n")
    f.write(f"训练样本: {len(train_df)}\n")
    f.write(f"验证样本: {len(val_df)}\n\n")
    f.write(f"{'='*60}\n")
    f.write(f"\n最终验证集评估结果:\n\n")
    for key, value in eval_results.items():
        if isinstance(value, float):
            f.write(f"  {key}: {value:.4f}\n")
logger.info(f"  评估报告已保存: {report_file}")

logger.info(f"\n所有结果已保存到: {results_dir}/")
logger.info("模型保存完成")