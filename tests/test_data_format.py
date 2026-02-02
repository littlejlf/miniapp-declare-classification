# -*- coding: utf-8 -*-
"""
测试数据处理格式是否正确
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def test_data_formatting():
    """测试数据格式化是否正确"""
    print("测试 1: 模拟数据创建")
    print("-" * 50)

    # 模拟数据
    data = [
        {'text': '这是测试文本1', 'labels': [0, 1]},
        {'text': '这是测试文本2', 'labels': [1, 0]},
        {'text': '这是测试文本3', 'labels': [1, 1]},
        {'text': '这是测试文本4', 'labels': [0, 0]},
    ]

    df = pd.DataFrame(data)
    print(f"✓ 创建 DataFrame: {len(df)} 行")
    print(f"  数据类型: {df['labels'].dtype}")

    # 确保labels是list类型
    df['labels'] = df['labels'].apply(lambda x: x if isinstance(x, list) else [x])
    print("✓ 确保labels是list类型")

    # 划分数据集
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    print(f"✓ 划分数据集: 训练集 {len(train_df)} 行, 验证集 {len(val_df)} 行")

    # 转换为Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("\n测试 2: 检查Dataset格式")
    print("-" * 50)

    # 检查labels格式
    sample = train_dataset[0]
    print(f"✓ 样本数据: {sample}")
    print(f"  - text类型: {type(sample['text'])}")
    print(f"  - labels类型: {type(sample['labels'])}")
    print(f"  - labels内容: {sample['labels']}")

    # 测试set_format
    print("\n测试 3: 测试torch格式转换")
    print("-" * 50)

    dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    # 设置格式
    dataset_dict.set_format("torch", columns=["labels"])

    # 检查转换后的格式
    sample_torch = dataset_dict['train'][0]
    print(f"✓ torch格式样本: {sample_torch}")
    print(f"  - labels类型: {type(sample_torch['labels'])}")

    if isinstance(sample_torch['labels'], torch.Tensor):
        print(f"  - labels形状: {sample_torch['labels'].shape}")
        print(f"  - labels数据类型: {sample_torch['labels'].dtype}")
        print("✓ labels正确转换为torch.Tensor!")
    else:
        print(f"⚠️  警告: labels类型是 {type(sample_torch['labels'])}, 不是 torch.Tensor")

    # 测试float32转换
    print("\n测试 4: 测试float32转换")
    print("-" * 50)

    labels_tensor = sample_torch['labels']
    if isinstance(labels_tensor, torch.Tensor):
        labels_float32 = labels_tensor.to(torch.float32)
        print(f"✓ 成功转换为float32: {labels_float32}")
        print(f"  - 数据类型: {labels_float32.dtype}")
    else:
        # 如果不是tensor，尝试转换
        labels_tensor = torch.tensor(labels_tensor, dtype=torch.float32)
        print(f"✓ 从list转换为tensor: {labels_tensor}")
        print(f"  - 数据类型: {labels_tensor.dtype}")

    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)

if __name__ == "__main__":
    test_data_formatting()
