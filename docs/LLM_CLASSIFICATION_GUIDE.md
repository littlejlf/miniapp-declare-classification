# LLM 分类器使用指南

本文档介绍如何使用三种 LLM 分类器对小程序隐私声明进行分类。

---

## 一、分类器概述

项目提供三种 LLM 分类器，每种分类器使用不同的提示词策略：

### 1. 统一分类器 (`classify_unified.py`)

- **提示词**: `prompts/classification_prompt.md`
- **特点**: 在一次调用中同时评估必要性违规和表述模糊违规
- **适用场景**: 需要同时获取两个维度的判断结果

### 2. 必要性独立分类器 (`classify_necessity.py`)

- **提示词**: `prompts/necessity_violation_prompt.md`
- **特点**: 专注于评估数据收集的"目的"与"手段"之间的逻辑问题
- **违规类型**:
  - 目的异常: 所声明的数据收集目的在逻辑上不成立
  - 可替代冗余: 存在功能等效但隐私侵害更小的替代方案
- **适用场景**: 只关注必要性维度的评估

### 3. 表述模糊独立分类器 (`classify_ambiguity.py`)

- **提示词**: `prompts/ambiguity_violation_prompt.md`
- **特点**: 专注于评估声明的文本表述是否清晰、具体
- **违规类型**:
  - 语义重复: 目的仅是对数据收集行为本身的同义转述
  - 场���缺失: 未指明具体业务流程或服务场景
  - 表述宽泛: 使用过于宽泛或模糊的表述
- **适用场景**: 只关注表述模糊维度的评估

---

## 二、环境配置

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 设置 API Key

使用阿里云 DashScope API，需要设置环境变量：

```bash
# Linux / macOS
export DASHSCOPE_API_KEY="your_api_key_here"

# Windows
set DASHSCOPE_API_KEY=your_api_key_here
```

或在 `.env` 文件中配置：

```bash
DASHSCOPE_API_KEY=your_api_key_here
```

### 2.3 选择模型

默认模型为 `qwen-plus`，可通过环境变量修改：

```bash
export LLM_MODEL_ID="qwen-plus"  # 或其他支持的模型
```

---

## 三、使用方式

### 3.1 单独运行分类器

#### 运行统一分类器

```bash
python experiments/llm_prompting/classify_unified.py
```

**输出文件**: `results/predictions/llm_unified_results.jsonl`

**输出格式**:
```json
{
  "statement": "目的声明文本",
  "result": "{\"has_necessity_violation\": false, \"has_ambiguity_violation\": true, \"reason\": \"...\"}",
  "status": "success"
}
```

#### 运行必要性独立分类器

```bash
python experiments/llm_prompting/classify_necessity.py
```

**输出文件**: `results/predictions/llm_necessity_results.jsonl`

**输出格式**:
```json
{
  "statement": "目的声明文本",
  "result": "{\"has_necessity_violation\": true, \"necessity_type\": \"目的异常\", \"reason\": \"...\"}",
  "status": "success"
}
```

#### 运行表述模糊独立分类器

```bash
python experiments/llm_prompting/classify_ambiguity.py
```

**输出文件**: `results/predictions/llm_ambiguity_results.jsonl`

**输出格式**:
```json
{
  "statement": "目的声明文本",
  "result": "{\"has_ambiguity_violation\": true, \"ambiguity_type\": \"场景缺失\", \"reason\": \"...\"}",
  "status": "success"
}
```

---

### 3.2 运行所有分类器（推荐）

使用 `run_all_classifications.py` 可以：

1. 并行运行三个分类器
2. 自动合并独立分类器的结果
3. 对比统一分类器和独立分类器的一致性

```bash
python experiments/llm_prompting/run_all_classifications.py
```

**输出文件**:
| 文件 | 说明 |
|------|------|
| `llm_unified_results.jsonl` | 统一分类器结果 |
| `llm_necessity_results.jsonl` | 必要性分类器结果 |
| `llm_ambiguity_results.jsonl` | 表述模糊分类器结果 |
| `llm_independent_merged.jsonl` | 独立分类器合并结果 |
| `classifier_comparison.json` | 分类器一致性对比报告 |

**合并结果格式** (`llm_independent_merged.jsonl`):
```json
{
  "statement": "目的声明文本",
  "necessity": {
    "has_violation": false,
    "type": null,
    "reason": "必要性分析理由...",
    "raw_response": "LLM原始返回"
  },
  "ambiguity": {
    "has_violation": true,
    "type": "场景缺失",
    "reason": "表述模糊分析理由...",
    "raw_response": "LLM原始返回"
  }
}
```

---

### 3.3 评估分类结果

使用 `evaluate_classification.py` 评估分类性能，需要有标签数据。

#### 评估统一分类器

```bash
python experiments/llm_prompting/evaluate_classification.py \
    --input results/predictions/llm_unified_results.jsonl \
    --labels data/raw/aggregate_datas_label.jsonl \
    --type unified
```

#### 评估独立分类器

```bash
python experiments/llm_prompting/evaluate_classification.py \
    --input results/predictions/llm_independent_merged.jsonl \
    --labels data/raw/aggregate_datas_label.jsonl \
    --type independent
```

#### 对比两种分类器

```bash
python experiments/llm_prompting/evaluate_classification.py \
    --input results/predictions/llm_independent_merged.jsonl \
    --labels data/raw/aggregate_datas_label.jsonl \
    --type compare
```

#### 评估所有分类器

```bash
python experiments/llm_prompting/evaluate_classification.py \
    --input results/predictions/llm_independent_merged.jsonl \
    --labels data/raw/aggregate_datas_label.jsonl \
    --type both
```

**评估报告输出**:
| 文件 | 说明 |
|------|------|
| `evaluation/unified_evaluation_report.json` | 统一分类器评估报告 |
| `evaluation/independent_evaluation_report.json` | 独立分类器评估报告 |
| `evaluation/classifier_comparison_report.json` | 性能对比报告 |

---

## 四、评估指标

### 4.1 基础指标

- **准确率 (Accuracy)**: 预测正确的比例
- **精确率 (Precision)**: 正类预测的精确度
- **召回率 (Recall)**: 正类召回的完整度
- **F1分数 (F1-Score)**: 精确率和召回率的调和平均

### 4.2 一致性指标

- **一致率**: 两个分类器预测一致的比例
- **Cohen's Kappa**: 两个分类器的一致性系数（考虑随机一致性）

### 4.3 混淆矩阵

显示真正例 (TP)、假正例 (FP)、假反例 (FN)、真反例 (TN) 的分布。

---

## 五、高级配置

### 5.1 修改输入文件

编辑各分类器脚本中的 `input_file` 变量：

```python
input_file = project_root / "data" / "raw" / "all_declares.jsonl"
```

### 5.2 调整并发数

修改 `concurrency` 参数（默认为 10）：

```python
await classify_from_file(
    input_file=input_file,
    output_file=output_file,
    concurrency=20,  # 增加并发数
    chunk_size=100
)
```

### 5.3 修改分块大小

修改 `chunk_size` 参数（默认为 100），用于进度追踪：

```python
chunk_size=50  # 每50条为一组
```

---

## 六、常见问题

### Q1: API 调用失败怎么办？

A: 检查以下几点：
1. API Key 是否正确设置
2. API 额度是否充足
3. 网络连接是否正常
4. 模型 ID 是否正确

### Q2: 如何提高处理速度？

A: 可以：
1. 增加并发数 (`concurrency`)
2. 减少温度参数 (`temperature`) 以加快响应
3. 使用更快的模型

### Q3: 结果中的 `status: format_error` 是什么意思？

A: 表示 LLM 返回的内容无法解析为 JSON，但原始响应仍保存在 `result` 字段中。

### Q4: 如何只处理部分数据？

A: 可以在输入文件中只保留需要处理的数据，或在脚本中添加数据切片逻辑。

---

## 七、输出数据结构

### 7.1 统一分类器输出

```json
{
  "statement": "为了用户登录，开发者收集你的昵称和头像",
  "result": {
    "has_necessity_violation": false,
    "has_ambiguity_violation": false,
    "reason": "【必要性分析】登录功能需要用户身份识别，收集昵称和头像合理...【表述模糊分析】声明明确了业务场景为登录，表述清晰..."
  },
  "status": "success"
}
```

### 7.2 独立分类器合并输出

```json
{
  "statement": "为了系统开发，开发者收集你的位置信息",
  "necessity": {
    "has_violation": true,
    "type": "目的异常",
    "reason": "系统开发不需要位置信息，属于目的异常",
    "raw_response": "{\"has_necessity_violation\": true, ...}"
  },
  "ambiguity": {
    "has_violation": false,
    "type": null,
    "reason": "声明明确说明了数据收集的目的",
    "raw_response": "{\"has_ambiguity_violation\": false, ...}"
  }
}
```

---

## 八、提示词自定义

如需修改提示词，编辑 `prompts/` 目录下的对应文件：

- `classification_prompt.md` - 统一分类器提示词
- `necessity_violation_prompt.md` - 必要性分类器提示词
- `ambiguity_violation_prompt.md` - 表述模糊分类器提示词

修改提示词后无需修改代码，重新运行分类器即可使用新提示词。

---

## 九、参考链接

- [阿里云 DashScope API 文档](https://help.aliyun.com/zh/dashscope/)
- [项目主 README](../README.md)
- [项目详细文档](./project_readme.md)
