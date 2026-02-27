# LLM分类实验使用说明

本目录包含LLM隐私声明分类的完整实验代码。

## 快速开始

### 1. 设置API Key

```bash
export DASHSCOPE_API_KEY='your-api-key-here'
```

### 2. 运行完整实验

```bash
# 方式1: 使用Python脚本（推荐）
cd /root/miniapp
python experiments/llm_prompting/run_llm_experiment.py

# 方式2: 使用Bash脚本
bash experiments/llm_prompting/run_llm_experiment.sh
```

### 3. 查看结果

实验完成后，结果保存在 `results/predictions/` 目录：

```bash
ls -lh results/predictions/
```

## 文件说明

### 核心脚本

| 文件 | 说明 |
|------|------|
| `run_llm_experiment.py` | 一键运行完整实验（推荐） |
| `run_llm_experiment.sh` | Bash版本的实验脚本 |
| `experiment_config.py` | 实验配置文件 |

### 分类器

| 文件 | 说明 |
|------|------|
| `classify_unified.py` | 统一分类器（同时判断两个维度） |
| `classify_necessity.py` | 必要性独立分类器 |
| `classify_ambiguity.py` | 表述模糊独立分类器 |

### 工具脚本

| 文件 | 说明 |
|------|------|
| `run_all_classifications.py` | 运行所有分类器并合并结果 |
| `evaluate_classification.py` | 评估分类器性能 |
| `sample_llm_results.py` | 按比例采样数据 |

## 高级用法

### 只运行部分步骤

```bash
# 只运行分类器
python run_llm_experiment.py --steps unified necessity ambiguity

# 只评估结果
python run_llm_experiment.py --steps eval-unified eval-independent compare

# 只采样数据
python run_llm_experiment.py --steps sample
```

### 指定采样数量

```bash
python run_llm_experiment.py --sample-size 100
```

### 使用不同的模型

修改 `experiment_config.py` 或设置环境变量：

```bash
export LLM_MODEL_ID="qwen-max"
python run_llm_experiment.py
```

### 更换提示词

1. 创建新的提示词文件，例如 `my_prompt.md`
2. 修改 `experiment_config.py` 中的提示词路径：

```python
UNIFIED_PROMPT_FILE = PROMPTS_DIR / "my_prompt.md"
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `llm_unified_results.jsonl` | 统一分类器的原始结果 |
| `llm_necessity_results.jsonl` | 必要性分类器的原始结果 |
| `llm_ambiguity_results.jsonl` | 表述模糊分类器的原始结果 |
| `llm_independent_merged.jsonl` | 独立分类器合并后的结果 |
| `llm_sampled_200.jsonl` | 采样的200条数据 |

### 评估报告

| 文件 | 说明 |
|------|------|
| `evaluation/llm_prompt_unified_evaluation_report.json` | 统一分类器评估报告 |
| `evaluation/independent_evaluation_report.json` | 独立分类器评估报告 |
| `evaluation/classifier_comparison_report.json` | 分类器对比报告 |

## 实验流程

```
1. 统一分类器
   └─> llm_unified_results.jsonl

2. 必要性分类器
   └─> llm_necessity_results.jsonl

3. 表述模糊分类器
   └─> llm_ambiguity_results.jsonl

4. 合并独立分类器结果
   └─> llm_independent_merged.jsonl

5. 评估统一分类器
   └─> evaluation/llm_prompt_unified_evaluation_report.json

6. 评估独立分类器
   └─> evaluation/independent_evaluation_report.json

7. 对比两种分类器
   └─> evaluation/classifier_comparison_report.json

8. 采样200条数据
   └─> llm_sampled_200.jsonl
```

## 常见问题

### Q: 如何查看API调用进度？

A: 分类器会打印实时进度信息，包括：
- 已处理/总数
- 成功/失败数量
- 预估剩余时间

### Q: 如何中断后恢复？

A: 使用分块处理模式，即使中断也能保留已完成的结果：

```python
# 在 classify_*.py 中
chunk_size=100  # 每100条保存一次
```

### Q: API调用失败怎么办？

A: 代码内置了自动重试机制（最多5次），如果仍然失败：
1. 检查API key是否正确
2. 检查网络连接
3. 检查API余额
4. 降低并发数：`export LLM_CONCURRENCY=5`

### Q: 如何使用其他LLM API？

A: 修改 `utils/llm_api.py`，添加新的API客户端。

## 费用估算

以阿里云Qwen-Plus为例：
- 输入：¥0.004/千tokens
- 输出：¥0.012/千tokens
- 1137条样本，平均每条约100 tokens输入，200 tokens输出
- 预估费用：约 ¥1-2

## 更新日志

- 2024-02-27: 初始版本，支持三种分类器和完整评估流程
