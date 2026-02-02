# -*- coding: utf-8 -*-
"""
全面检查项目中的语法错误和常见问题
"""
import os
import ast
import re
from pathlib import Path

def check_file(filepath):
    """检查单个文件的问题"""
    issues = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # 语法检查
        try:
            tree = ast.parse(content, filename=filepath)
        except SyntaxError as e:
            issues.append(f"❌ 语法错误: {e}")
            return issues

        # 检查常见问题
        for i, line in enumerate(lines, 1):
            # 检查 .to() 方法调用
            if '.to(' in line and 'lambda' in line:
                issues.append(f"⚠️  行 {i}: lambda 函数中使用 .to() 方法可能导致类型错误")

            # 检查可能的多标签分类问题
            if 'problem_type="multi_label"' in line or 'problem_type="multi_label_classification"' in line:
                issues.append(f"✓ 行 {i}: 使用了多标签分类设置")

            # 检查 DataCollatorWithPadding
            if 'DataCollatorWithPadding' in line:
                if 'class MultiLabelDataCollator' in content:
                    issues.append(f"✓ 行 {i}: 使用了自定义多标签DataCollator")
                else:
                    issues.append(f"⚠️  行 {i}: 使用标准DataCollator，多标签分类可能需要自定义")

            # 检查 set_format("torch")
            if 'set_format("torch")' in line or "set_format('torch')" in line:
                issues.append(f"✓ 行 {i}: 设置了torch格式")

            # 检查 labels 类型处理
            if '.float()' in line and 'labels' in line:
                issues.append(f"✓ 行 {i}: 将labels转换为float类型")

        return issues

    except Exception as e:
        return [f"❌ 读取文件失败: {e}"]

def main():
    project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("🔍 全面检查项目代码问题")
    print("=" * 70)

    # 检查关键文件
    files_to_check = [
        'experiments/baseline/train_roberta.py',
        'experiments/llm_prompting/classify_async.py',
        'data_processing/augment_data.py',
        'utils/llm_api.py',
        'utils/config.py',
        'quickstart.py',
    ]

    total_issues = 0

    for filepath in files_to_check:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"\n⚠️  文件不存在: {filepath}")
            continue

        print(f"\n{'─' * 70}")
        print(f"📄 检查文件: {filepath}")
        print(f"{'─' * 70}")

        issues = check_file(full_path)

        if issues:
            for issue in issues:
                print(f"  {issue}")
                if issue.startswith('❌'):
                    total_issues += 1
        else:
            print("  ✅ 未发现问题")

    print(f"\n{'=' * 70}")
    if total_issues == 0:
        print("✅ 所有文件检查通过！")
    else:
        print(f"⚠️  发现 {total_issues} 个严重问题")
    print("=" * 70)

if __name__ == "__main__":
    main()
