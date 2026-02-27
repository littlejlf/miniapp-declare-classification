#!/bin/bash
# -*- coding: utf-8 -*-
# LLM分类实���一键运行脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "LLM分类实验启动"
echo "========================================"

# 设置项目根目录
PROJECT_ROOT="/root/miniapp"
cd "$PROJECT_ROOT"

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
fi

# 检查API key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "错误: 请设置环境变量 DASHSCOPE_API_KEY"
    echo "示例: export DASHSCOPE_API_KEY='your-api-key'"
    exit 1
fi

echo "API Key已设置: ${DASHSCOPE_API_KEY:0:10}..."
echo ""

# 设置路径
INPUT_FILE="$PROJECT_ROOT/data/raw/aggregate_datas_label.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/results/predictions"
LABEL_FILE="$INPUT_FILE"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "步骤1: 运行统一分类器"
echo "========================================"
python experiments/llm_prompting/classify_unified.py

echo ""
echo "========================================"
echo "步骤2: 运行必要性独立分类器"
echo "========================================"
python experiments/llm_prompting/classify_necessity.py

echo ""
echo "========================================"
echo "步骤3: 运行表述模糊独立分类器"
echo "========================================"
python experiments/llm_prompting/classify_ambiguity.py

echo ""
echo "========================================"
echo "步骤4: 合并独立分类器结果"
echo "========================================"
python experiments/llm_prompting/run_all_classifications.py

echo ""
echo "========================================"
echo "步骤5: 评估统一分类器结果"
echo "========================================"
python experiments/llm_prompting/evaluate_classification.py \
    --input "$OUTPUT_DIR/llm_unified_results.jsonl" \
    --labels "$LABEL_FILE" \
    --type unified \
    --output "$OUTPUT_DIR/evaluation"

echo ""
echo "========================================"
echo "步骤6: 评估独立分类器结果"
echo "========================================"
python experiments/llm_prompting/evaluate_classification.py \
    --input "$OUTPUT_DIR/llm_independent_merged.jsonl" \
    --labels "$LABEL_FILE" \
    --type independent \
    --output "$OUTPUT_DIR/evaluation"

echo ""
echo "========================================"
echo "步骤7: 对比两种分类器"
echo "========================================"
python experiments/llm_prompting/evaluate_classification.py \
    --input "$OUTPUT_DIR/llm_independent_merged.jsonl" \
    --labels "$LABEL_FILE" \
    --type compare \
    --output "$OUTPUT_DIR/evaluation"

echo ""
echo "========================================"
echo "步骤8: 采样200条数据（1:1:1:1比例）"
echo "========================================"
python experiments/llm_prompting/sample_llm_results.py \
    --input "$OUTPUT_DIR/llm_unified_results.jsonl" \
    --output "$OUTPUT_DIR/llm_sampled_200.jsonl" \
    --size 200 \
    --seed 42

echo ""
echo "========================================"
echo "实验完成！"
echo "========================================"
echo "结果文件位置: $OUTPUT_DIR"
echo ""
echo "生成文件列表:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "评估报告位置: $OUTPUT_DIR/evaluation"
ls -lh "$OUTPUT_DIR/evaluation" 2>/dev/null || echo "评估目录尚未创建"
