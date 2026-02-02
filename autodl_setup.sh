#!/bin/bash
# AutoDL 环境设置脚本（优化版）

set -e  # 遇到错误立即退出

echo "========================================="
echo "Miniapp Privacy Classification - AutoDL Setup"
echo "========================================="

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查Python版本
echo -e "\n${YELLOW}[1/6] 检查Python版本...${NC}"
python --version
PYTHON_VERSION=$(python --version | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"

# 2. 检查预装的PyTorch
echo -e "\n${YELLOW}[2/6] 检查预装的PyTorch...${NC}"
if python -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}✓ PyTorch 版本: $PYTORCH_VERSION${NC}"
    echo -e "${GREEN}✓ CUDA 版本: $CUDA_VERSION${NC}"
    echo -e "${GREEN}✓ GPU 可用: $(python -c "import torch; print(torch.cuda.is_available())")${NC}"
else
    echo -e "${YELLOW}⚠ 未检测到PyTorch，可能需要安装${NC}"
fi

# 3. 检查是否已创建虚拟环境
echo -e "\n${YELLOW}[3/6] 检查虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ 虚拟环境已存在${NC}"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧的虚拟环境..."
        rm -rf venv
    else
        echo "使用现有虚拟环境"
        USE_VENV=1
    fi
fi

# 4. 创建虚拟环境（如果需要）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python -m venv venv
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
    USE_VENV=1
fi

# 5. 激活虚拟环境并升级pip
echo -e "\n${YELLOW}[4/6] 激活虚拟环境并升级pip...${NC}"
source venv/bin/activate
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo -e "${GREEN}✓ pip 已升级${NC}"

# 6. 安装依赖
echo -e "\n${YELLOW}[5/6] 安装项目依赖...${NC}"

# 检查是否使用PyTorch基础镜像
read -p "是否使用PyTorch基础镜像? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    # 不使用PyTorch镜像，安装所有依赖
    echo "安装完整依赖（包括PyTorch）..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    # 使用PyTorch镜像，跳过torch安装
    echo "使用AutoDL优化依赖（跳过已预装的PyTorch）..."
    pip install -r requirements-autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo -e "${GREEN}✓ 依赖安装完成${NC}"

# 7. 设置环境变量和软链接
echo -e "\n${YELLOW}[6/6] 设置持久化存储...${NC}"

# 创建持久化存储目录
mkdir -p ~/autodl-fs/miniapp-models
mkdir -p ~/autodl-fs/miniapp-predictions
mkdir -p ~/autodl-fs/miniapp-logs
mkdir -p ~/autodl-fs/cache

# 创建软链接
rm -rf results/models results/predictions results/logs 2>/dev/null || true
ln -s ~/autodl-fs/miniapp-models results/models
ln -s ~/autodl-fs/miniapp-predictions results/predictions
ln -s ~/autodl-fs/miniapp-logs results/logs

# 设置缓存环境变量
echo '' >> ~/.bashrc
echo '# Miniapp 项目环境变量' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-fs/cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-fs/cache' >> ~/.bashrc

echo -e "${GREEN}✓ 持久化存储设置完成${NC}"

# 完成
echo ""
echo "========================================="
echo -e "${GREEN}✓ 安装完成！${NC}"
echo "========================================="
echo ""
echo "已安装内容："
echo "  - Python: $PYTHON_VERSION"
echo "  - 虚拟环境: ./venv/"
echo "  - 项目依赖: 已安装"
echo ""
echo "使用方法："
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 设置API Key: export DASHSCOPE_API_KEY='your-key-here'"
echo "  3. 运行实验: bash run_experiment.sh"
echo "     或: python quickstart.py baseline"
echo ""
echo "下一步："
echo "  请设置 DASHSCOPE_API_KEY 环境变量后开始实验"
echo ""
