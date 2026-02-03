# -*- coding: utf-8 -*-
"""
必要性违规分类模型训练脚本

单独训练一个BERT模型来识别必要性违规（目的异常、可替代冗余）。
使用类别权重处理数据不平衡问题。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到���径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import numpy as np
import json
from collections import Counter
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from utils.logger import get_logger
from utils.metrics import format_metrics_report

# ==================== 配置参数 ====================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
DATA_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"
OUTPUT_DIR = project_root / "results" / "models" / "roberta_necessity_classifier"

# 训练参数
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# 类别权重配置
# 自动计算权重或手动指定
# 如果 MANUAL_WEIGHTS 为 None，则根据数据分布自动计算
MANUAL_WEIGHTS = None  # 例如: [1.0, 2.0] 表示正类权重是负类的2倍
# 或者使用 'balanced' 自动计算

logger = get_logger(__name__)


class WeightedTrainer(Trainer):
    """
    带类别权重的Trainer
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算加权损失
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # 获取logits
        logits = outputs.get("logits")

        # 如果有类别权重，使用加权损失
        if self.class_weights is not None:
            # 将权重移到正确的设备
            class_weights = self.class_weights.to(logits.device)

            # 计算加权交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # 使用默认的交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def calculate_class_weights(labels, method='balanced'):
    """
    计算类别权重

    Args:
        labels: 标签数组
        method: 'balanced' 或 'inverse'

    Returns:
        class_weights: Tensor, 形状为 (num_classes,)
    """
    # 统计每个类别的样本数
    label_counts = Counter(labels)
    num_classes = len(label_counts)
    total_samples = len(labels)

    logger.info(f"类别分布统计:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  类别 {label}: {count} 样本 ({count/total_samples*100:.2f}%)")

    if method == 'balanced':
        # sklearn的balanced方式: n_samples / (n_classes * n_samples_per_class)
        class_weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)
    elif method == 'inverse':
        # 简单的反比方式
        class_weights = []
        max_count = max(label_counts.values())
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weight = max_count / count
            class_weights.append(weight)
    else:
        raise ValueError(f"未知的权重计算方法: {method}")

    class_weights = torch.tensor(class_weights, dtype=torch.float)

    logger.info(f"计算得到的类别权重: {class_weights.tolist()}")
    return class_weights


def load_and_prepare_data(label_index=0):
    """
    加载并准备数据

    Args:
        label_index: 0=必要性, 1=表述模糊
    """
    logger.info(f"加载数据从: {DATA_FILE}")

    # 从JSONL文件加载数据
    data = []
    label_counts = Counter()

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                labels = item.get('label', [0, 0])
                # 只取指定维度的标签
                single_label = labels[label_index]
                data.append({
                    'text': item.get('statement', ''),
                    'labels': single_label
                })
                label_counts[single_label] += 1

    df = pd.DataFrame(data)
    logger.info(f"加载数据完成，共 {len(df)} 条")
    logger.info(f"标签分布: {dict(label_counts)}")

    # 划分训练集和验证集（使用分层采样保持标签分布）
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['labels']  # 分层采样
    )

    logger.info(f"训练集: {len(train_df)} 条")
    logger.info(f"验证集: {len(val_df)} 条")

    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    return dataset_dict, train_df['labels'].values


def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    logits = pred.predictions
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """主训练函数"""
    logger.info(f"{'='*60}")
    logger.info(f"开始训练必要性违规分类模型")
    logger.info(f"{'='*60}")

    # 1. 准备数据
    dataset_dict, train_labels = load_and_prepare_data(label_index=0)  # 0=必要性

    # 2. 计算类别权重
    if MANUAL_WEIGHTS is not None:
        logger.info(f"使用手动指定的权重: {MANUAL_WEIGHTS}")
        class_weights = torch.tensor(MANUAL_WEIGHTS, dtype=torch.float)
    else:
        logger.info("自动计算类别权重 (balanced)...")
        class_weights = calculate_class_weights(train_labels, method='balanced')

    # 3. 初始化Tokenizer和模型
    logger.info(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # 二分类
    )

    # 4. Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    logger.info("Tokenization...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # 移除不需要的列
    columns_to_remove = [
        col for col in tokenized_datasets["train"].column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    tokenized_datasets.set_format("torch")

    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        save_total_limit=1,  # 只保留最佳模型
        report_to=None,  # 不使用wandb
    )

    # 6. 初始化Trainer（使用带权重的版本）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    # 7. 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 8. 评估
    logger.info("训练完成，评估最佳模型...")
    eval_results = trainer.evaluate()

    logger.info(f"\n{'='*60}")
    logger.info(f"最终评估结果")
    logger.info(f"{'='*60}")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")

    # 9. 详细评估报告
    logger.info(f"\n{'='*60}")
    logger.info(f"详细分类报告")
    logger.info(f"{'='*60}")

    # 获取预测结果
    raw_pred, _, _ = trainer.predict(tokenized_datasets["validation"])
    y_pred = np.argmax(raw_pred, axis=1)
    y_true = tokenized_datasets["validation"]["labels"].numpy()

    # 打印分类报告
    print("\n" + classification_report(
        y_true, y_pred,
        target_names=['正常', '违规'],
        digits=4
    ))

    # 打印混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # 10. 保存模型和配置
    logger.info(f"\n保存模型到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存配置信息
    config = {
        'model_name': MODEL_NAME,
        'task': 'necessity_classification',
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'class_weights': class_weights.tolist(),
        'eval_results': {k: v for k, v in eval_results.items() if isinstance(v, float)}
    }

    config_file = OUTPUT_DIR / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"配置已保存到: {config_file}")
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
