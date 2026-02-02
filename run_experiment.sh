#!/bin/bash
# AutoDL 实验运行脚本

# 激活虚拟环境
source venv/bin/activate

# 加载环境变量（如果存在）
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 检查必需的环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "错误: 请设置 DASHSCOPE_API_KEY 环境变量"
    echo "可以通过以下方式设置:"
    echo "  export DASHSCOPE_API_KEY='your-key-here'"
    echo "或者创建 .env 文件（参考 .env.template）"
    exit 1
fi

# 显示菜单
show_menu() {
    echo ""
    echo "========================================="
    echo "Miniapp Privacy Classification - 实验菜单"
    echo "========================================="
    echo "1. 训练 RoBERTa 基线模型"
    echo "2. 运行 LLM 提示词分类"
    echo "3. 运行数据增强"
    echo "4. 退出"
    echo "========================================="
}

# 主循环
while true; do
    show_menu
    read -p "请选择实验 [1-4]: " choice

    case $choice in
        1)
            echo ""
            echo "开始训练 RoBERTa 基线模型..."
            python experiments/baseline/train_roberta.py
            ;;
        2)
            echo ""
            echo "开始 LLM 提示词分类..."
            python experiments/llm_prompting/classify_async.py
            ;;
        3)
            echo ""
            echo "开始数据增强..."
            python data_processing/augment_data.py
            ;;
        4)
            echo "退出"
            break
            ;;
        *)
            echo "无效选择，请重新输入"
            ;;
    esac
done
