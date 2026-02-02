# AutoDL 目录组织指南

## 推荐的项目位置

```bash
/root/miniapp-declare-classification/    # ✅ 项目代码（推荐位置）
```

### 完整的目录结构

```
/root/
├── miniapp-declare-classification/      # ✅ 你的项目代码放这里
│   ├── data/                           # 小数据集（<1GB）
│   │   ├── raw/
│   │   │   ├── all_declares.jsonl      # 4MB
│   │   │   └── aggregate_datas_label.jsonl  # 152KB
│   │   ├── processed/
│   │   └── augmented/
│   │
│   ├── experiments/                     # 实验代码
│   │   ├── baseline/
│   │   ├── llm_prompting/
│   │   ├── stf_enhanced/
│   │   └── distillation/
│   │
│   ├── results/                        # 训练结果（软链接到持久化存储）
│   │   ├── models/                     # -> /root/autodl-fs/miniapp-models/
│   │   ├── predictions/                # -> /root/autodl-fs/miniapp-predictions/
│   │   └── logs/                       # -> /root/autodl-fs/miniapp-logs/
│   │
│   ├── venv/                           # 虚拟环境
│   ├── configs/                        # 配置文件
│   ├── utils/                          # 工具模块
│   ├── prompts/                        # 提示词
│   ├── autodl_setup.sh                 # 设置脚本
│   ├── run_experiment.sh               # 运行脚本
│   ├── quickstart.py                   # 快速启动
│   └── README.md
│
├── autodl-fs/                          # ✅ 持久化存储（重启后保留）
│   ├── miniapp-models/                 # 训练好的模型
│   ├── miniapp-predictions/            # 预测结果
│   ├── miniapp-logs/                   # 日志文件
│   └── cache/                          # Transformers缓存
│
├── auto-dl-tmp/                        # ⚡ 临时高性能存储
│   └── tmp/                            # 临时文件
│
└── workspace/                          # 其他项目
```

## 快速设置命令

```bash
# SSH连接到AutoDL
ssh root@connect.autodl.com -p <端口>

# 1. 确认当前位置
cd ~
pwd  # 应该显示 /root

# 2. 克隆项目
git clone https://github.com/littlejlf/miniapp-declare-classification.git

# 3. 进入项目目录
cd ~/miniapp-declare-classification

# 4. 创建持久化存储目录
mkdir -p ~/autodl-fs/miniapp-models
mkdir -p ~/autodl-fs/miniapp-predictions
mkdir -p ~/autodl-fs/miniapp-logs
mkdir -p ~/autodl-fs/cache

# 5. 创建软链接（将结果保存到持久化存储）
ln -s ~/autodl-fs/miniapp-models ~/miniapp-declare-classification/results/models
ln -s ~/autodl-fs/miniapp-predictions ~/miniapp-declare-classification/results/predictions
ln -s ~/autodl-fs/miniapp-logs ~/miniapp-declare-classification/results/logs

# 6. 设置环境变量
echo 'export HF_HOME=/root/autodl-fs/cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-fs/cache' >> ~/.bashrc
source ~/.bashrc

# 7. 安装依赖
bash autodl_setup.sh
```

## 为什么这样组织？

### ✅ `/root/miniapp-declare-classification/` - 项目代码
- **优点**：易于访问、符合习惯
- **适合**：代码、配置、小数据
- **注意**：实例关闭后可能清空

### ✅ `/root/autodl-fs/` - 持久化存储
- **优点**：实例关闭后**数据保留**
- **适合**：训练好的模型、预测结果、日志
- **必须**：重要数据必须放这里

### ⚡ `/root/auto-dl-tmp/` - 临时存储
- **优点**：读写速度快
- **适合**：训练时的临时文件、缓存
- **注意**：可能被清空

## 检查磁盘空间

```bash
# 查看各目录使用情况
df -h

# 查看项目大小
du -sh ~/miniapp-declare-classification

# 查看持久化存储使用情况
du -sh ~/autodl-fs/*
```

## 自动化设置脚本

创建 `~/setup_autodl.sh`：

```bash
#!/bin/bash
set -e

echo "设置AutoDL项目目录..."

# 创建持久化存储目录
mkdir -p ~/autodl-fs/miniapp-models
mkdir -p ~/autodl-fs/miniapp-predictions
mkdir -p ~/autodl-fs/miniapp-logs
mkdir -p ~/autodl-fs/cache

# 创建软链接
cd ~/miniapp-declare-classification
rm -rf results/models results/predictions results/logs
ln -s ~/autodl-fs/miniapp-models results/models
ln -s ~/autodl-fs/miniapp-predictions results/predictions
ln -s ~/autodl-fs/miniapp-logs results/logs

# 设置缓存目录
echo 'export HF_HOME=/root/autodl-fs/cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-fs/cache' >> ~/.bashrc

echo "✅ 设置完成！"
echo "项目目录: ~/miniapp-declare-classification"
echo "模型保存: ~/autodl-fs/miniapp-models"
echo "结果保存: ~/autodl-fs/miniapp-predictions"
```

使用方法：
```bash
chmod +x ~/setup_autodl.sh
~/setup_autodl.sh
```

## 验证设置

```bash
# 验证软链接
ls -la ~/miniapp-declare-classification/results/

# 应该看到：
# models -> /root/autodl-fs/miniapp-models
# predictions -> /root/autodl-fs/miniapp-predictions
# logs -> /root/autodl-fs/miniapp-logs

# 验证环境变量
echo $HF_HOME
# 应该显示：/root/autodl-fs/cache
```

## 数据安全检查清单

- [ ] 重要数据放在 `/root/autodl-fs/`（持久化存储）
- [ ] 创建软链接到项目目录
- [ ] 定期下载模型到本地
- [ ] 重要结果上传到GitHub或对象存储
- [ ] 关机前确认数据已保存

## 常见问题

### Q: 项目代码会丢失吗？
A: 实例关闭后 `/root/` 下的内容可能会清空，建议：
1. 代码推送到GitHub
2. 重要配置保存到 `autodl-fs/`

### Q: 如何备份数据？
A:
```bash
# 压缩并下载到本地
cd ~/autodl-fs
tar -czf backup.tar.gz miniapp-models/ miniapp-predictions/

# 通过AutoDL Web界面下载
# 或使用 scp 下载到本地
```

### Q: 磁盘空间不足怎么办？
A:
```bash
# 清理缓存
rm -rf ~/.cache/huggingface/
rm -rf ~/autodl-fs/cache/*

# 清理旧的模型
rm -rf ~/autodl-fs/miniapp-models/old_*/
```

## 总结

```bash
# ✅ 正确位置
代码:    ~/miniapp-declare-classification/
数据:    ~/miniapp-declare-classification/data/
模型:    ~/autodl-fs/miniapp-models/ (软链接到 results/models)
结果:    ~/autodl-fs/miniapp-predictions/ (软链接到 results/predictions)
日志:    ~/autodl-fs/miniapp-logs/ (软链接到 results/logs)
```

这样可以确保：
1. 代码易于访问和管理
2. 重要数据持久保存
3. 实例重启后不丢失训练成果
