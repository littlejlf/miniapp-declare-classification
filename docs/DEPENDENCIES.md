# 项目依赖检查清单

## ✅ 完整依赖列表

### 核心依赖（必需）

| 包名 | 版本要求 | 用途 | 使用位置 |
|------|---------|------|---------|
| **torch** | >=1.10.0 | PyTorch深度学习框架 | `train_roberta.py` |
| **transformers** | >=4.20.0 | Hugging Face Transformer模型 | `train_roberta.py` |
| **datasets** | >=2.0.0 | Hugging Face 数据集处理 | `train_roberta.py` ⚠️ |
| **dashscope** | >=1.10.0 | 阿里云LLM API | `classify_async.py`, `augment_data.py` |
| **scikit-learn** | >=1.0.0 | 机器学习评估指标 | `train_roberta.py`, `metrics.py` |
| **pandas** | >=1.3.0 | 数据处理 | `train_roberta.py` |
| **numpy** | >=1.21.0 | 数值计算 | `metrics.py` |
| **pyyaml** | >=5.4.0 | YAML配置文件解析 | `config.py` |
| **backoff** | >=2.1.0 | API重试机制 | `llm_api.py` |

### 工具依赖（推荐）

| 包名 | 版本要求 | 用途 | 使用位置 |
|------|---------|------|---------|
| **tqdm** | >=4.62.0 | 进度条显示 | 训练过程 |
| **matplotlib** | >=3.4.0 | 数据可视化 | `analysis/` |
| **seaborn** | >=0.11.0 | 统计可视化 | `analysis/` |

### 开发依赖（可选）

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| **jupyter** | >=1.0.0 | Jupyter笔记本 |
| **ipykernel** | >=6.0.0 | Jupyter内核 |

### Python 标准库（无需安装）

```
argparse, asyncio, datetime, enum, json, logging,
os, pathlib, platform, sys, typing
```

## ⚠️ 常见缺失包问题

### 问题 1: ModuleNotFoundError: No module named 'datasets'

**原因**: `train_roberta.py` 使用了 `datasets.Dataset`

**解决方案**:
```bash
pip install datasets>=2.0.0
```

### 问题 2: ModuleNotFoundError: No module named 'dashscope'

**原因**: LLM API 调用需要 `dashscope`

**解决方案**:
```bash
pip install dashscope>=1.10.0
```

### 问题 3: ModuleNotFoundError: No module named 'yaml'

**原因**: 配置文件读取需要 `pyyaml`

**解决方案**:
```bash
pip install pyyaml>=5.4.0
```

## 🔍 依赖检查命令

### 检查所有依赖是否安装

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查核心依赖
python -c "
import sys
packages = ['torch', 'transformers', 'datasets', 'dashscope',
            'sklearn', 'pandas', 'numpy', 'yaml', 'backoff']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} (缺失)')
        missing.append(pkg)

if missing:
    print(f'\n缺失的包: {missing}')
    sys.exit(1)
else:
    print('\n所有依赖都已安装！')
"
```

### 检查版本兼容性

```bash
# 检查 PyTorch 和 CUDA 兼容性
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

# 检查 Transformers 版本
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 📦 完整安装命令

### AutoDL 环境（推荐）

```bash
# 使用优化版依赖文件（跳过已安装的 PyTorch）
source venv/bin/activate
pip install -r requirements-autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 普通环境

```bash
# 安装所有依赖（包括 PyTorch）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 快速修复（仅安装缺失的包）

```bash
# 如果遇到缺失包错误，运行这个命令
pip install datasets dashscope pyyaml backoff scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 🔧 依赖版本兼容性

### 测试过的版本组合

| 包名 | 版本 | 状态 |
|------|------|------|
| Python | 3.8 ~ 3.11 | ✅ 兼容 |
| PyTorch | 1.10.0 ~ 2.3.0 | ✅ 兼容 |
| Transformers | 4.20.0 ~ 4.40.0 | ✅ 兼容 |
| Datasets | 2.0.0 ~ 2.18.0 | ✅ 兼容 |
| CUDA | 11.3 ~ 12.1 | ✅ 兼容 |

### 潜在冲突

| 包名 | 冲突说明 | 解决方案 |
|------|---------|---------|
| transformers | 版本过低可能不兼容某些模型 | 使用 >=4.20.0 |
| torch | 不同CUDA版本需要不同的torch | 根据CUDA版本选择 |

## 📝 依赖更新记录

### 2024-02-02
- ✅ 添加缺失的 `datasets>=2.0.0`
- ✅ 添加注释说明每个包的用途
- ✅ 创建 `requirements-autodl.txt` 优化版

### 注意事项

1. **datasets 包是必需的** - RoBERTa 训练脚本使用
2. **dashscope 需要 API Key** - 使用前必须设置环境变量
3. **PyTorch 版本要与 CUDA 匹配** - 否则 GPU 不可用
