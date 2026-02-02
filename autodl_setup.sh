#!/bin/bash
# AutoDL 环境设置脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "Miniapp Privacy Classification - AutoDL Setup"
echo "========================================="

# 1. 检查Python版本
echo "[1/5] 检查Python版本..."
python --version

# 2. 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "[2/5] 创建虚拟环境..."
    python -m venv venv
else
    echo "[2/5] 虚拟环境已存在"
fi

# 3. 激活虚拟环境
echo "[3/5] 激活虚拟环境..."
source venv/bin/activate

# 4. 升级pip
echo "[4/5] 升级pip..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装依赖
echo "[5/5] 安装项目依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "========================================="
echo "安装完成！"
echo "========================================="
echo ""
echo "使用方法:"
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 设置API Key: export DASHSCOPE_API_KEY='your-key-here'"
echo "  3. 运行实验: python experiments/baseline/train_roberta.py"
echo ""
