# AutoDL 使用指南

本项目已适配AutoDL云GPU平台，可以方便地在云端进行训练和实验。

## 快速开始

### 1. 上传项目到AutoDL

在AutoDL中打开Jupyter，使用以下命令克隆或上传项目：

```bash
# 如果项目在Git仓库
git clone <your-repo-url>

# 或者通过本地上传
# 1. 将项目打包为 zip
# 2. 在AutoDL的Jupyter中上传
# 3. 解压: unzip miniapp-declare-classification.zip
```

### 2. 设置环境

运行自动设置脚本：

```bash
cd miniapp-declare-classification
bash autodl_setup.sh
```

### 3. 配置API Key

创建 `.env` 文件并设置你的API Key：

```bash
# 复制模板
cp .env.template .env

# 编辑文件，填入你的API Key
vim .env
```

或通过命令行设置：

```bash
export DASHSCOPE_API_KEY='your-api-key-here'
```

### 4. 运行实验

使用交互式菜单：

```bash
bash run_experiment.sh
```

或直接运行Python脚本：

```bash
# 激活虚拟环境
source venv/bin/activate

# 训练基线模型
python experiments/baseline/train_roberta.py

# LLM分类
python experiments/llm_prompting/classify_async.py

# 数据增强
python data_processing/augment_data.py
```

## 目录结构说明

```
miniapp-declare-classification/
├── autodl_setup.sh           # AutoDL环境设置脚本
├── run_experiment.sh         # 实验运行脚本
├── .env.template            # 环境变量模板
├── requirements.txt         # Python依赖
├── data/                    # 数据目录
├── experiments/             # 实验代码
├── results/                 # 结果输出
└── ...
```

## GPU使用建议

### 选择合适的GPU镜像

- **基础训练**：RTX 3090 (24GB) 或 RTX 4090 (24GB)
- **大规模训练**：A100 (40GB/80GB)
- **LLM推理**：可以使用较小显存的GPU

### 显存优化

如果遇到显存不足：

1. 减小batch size（在 `configs/training_config.yaml` 中修改）
2. 使用梯度累积
3. 减小max_length参数

## 常见问题

### Q1: 如何查看GPU使用情况？

```bash
nvidia-smi
```

### Q2: 如何在后台运行训练？

```bash
nohup python experiments/baseline/train_roberta.py > train.log 2>&1 &
```

### Q3: 如何中断正在运行的训练？

```bash
# 查找进程
ps aux | grep python

# 终止进程
kill <pid>
```

### Q4: 训练中断后如何继续？

使用checkpoint恢复训练（在训练脚本中配置 `load_best_model_at_end=True`）

### Q5: 数据下载慢怎么办？

使用国内镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据存储建议

AutoDL的本地存储在实例关闭后会清空，建议：

1. **重要数据**：定期下载到本地或上传到OSS
2. **模型文件**：保存到云端存储（如AutoDL的持久化磁盘）
3. **日志文件**：实时查看或保存到持久化存储

## 下载结果

### 使用AutoDL的Web界面

1. 在Jupyter文件浏览器中找到 `results/` 目录
2. 选择需要下载的文件
3. 点击下载按钮

### 使用命令行压缩

```bash
# 压缩结果目录
cd results
tar -czf results_$(date +%Y%m%d).tar.gz models/ predictions/ logs/

# 下载压缩包（通过Web界面）
```

## 网络问题

如果遇到网络问题：

```bash
# 设置代理（如果AutoDL提供）
export http_proxy=http://<proxy>:<port>
export https_proxy=http://<proxy>:<port>

# 或者使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 成本优化建议

1. **选择合适的GPU**：不要过度配置
2. **及时关机**：实验完成后关闭实例
3. **使用 spot 实例**：如果AutoDL提供（更便宜但可能被中断）
4. **批量处理**：合并多个实验一起运行

## 联系支持

如果遇到问题：
1. 查看AutoDL官方文档
2. 检查项目的Issue
3. 提交新的Issue
