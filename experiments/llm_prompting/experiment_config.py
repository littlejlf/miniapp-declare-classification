# -*- coding: utf-8 -*-
"""
LLM分类实验配置文件

修改此文件可以更换：
1. 使用的模型
2. 提示词文件
3. 实验参数
4. 输出路径等
"""

import os
from pathlib import Path

# ==================== 项目路径 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ==================== 模型配置 ====================
# 可选模型（根据你使用的API服务商）:
# - 阿里云DashScope: qwen-plus, qwen-turbo, qwen-max
# - 其他模型可以在代码中添加支持

MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")  # 默认使用qwen-plus

# API配置
API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取
API_BASE_URL = os.getenv("API_BASE_URL", None)  # 可选：自定义API端点

# ==================== 提示词配置 ====================
# 提示词文件路径
PROMPTS_DIR = PROJECT_ROOT / "prompts"

UNIFIED_PROMPT_FILE = PROMPTS_DIR / "classification_prompt.md"
NECESSITY_PROMPT_FILE = PROMPTS_DIR / "necessity_violation_prompt.md"
AMBIGUITY_PROMPT_FILE = PROMPTS_DIR / "ambiguity_violation_prompt.md"

# 如果你想使用不同的提示词，可以修改上面的路径
# 例如：
# UNIFIED_PROMPT_FILE = PROMPTS_DIR / "my_custom_prompt.md"

# ==================== 模型参数配置 ====================
# 温度参数：控制输出的随机性
# - 0.0: 完全确定性输出，适合分类任务
# - 0.5: 适中的随机性
# - 1.0: 高随机性，适合创意任务
TEMPERATURE = 0.1

# 最大输出token数
MAX_TOKENS = 2000

# Top-p采样参数
TOP_P = 0.9

# ==================== 并发配置 ====================
# 并发请求数量
# - 数值越大，速度越快，但可能触发API限流
# - 建议值：5-20，根据API服务商的限制调整
CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))

# 分块大小
# - 将大批量任务分成小块处理，便于断点续传
# - 建议值：50-200
CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "100"))

# ==================== 数据配置 ====================
# 输入数据文件
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "aggregate_datas_label.jsonl"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "results" / "predictions"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

# 输出文件名
UNIFIED_OUTPUT_FILE = "llm_unified_results.jsonl"
NECESSITY_OUTPUT_FILE = "llm_necessity_results.jsonl"
AMBIGUITY_OUTPUT_FILE = "llm_ambiguity_results.jsonl"
INDEPENDENT_MERGED_FILE = "llm_independent_merged.jsonl"

# ==================== 采样配置 ====================
# 采样数量
SAMPLE_SIZE = 200

# 采样比例（四种组合：1:1:1:1）
# - (正常, 清晰)
# - (正常, 模糊)
# - (违规, 清晰)
# - (违规, 模糊)
SAMPLE_RATIO = [1, 1, 1, 1]

# 随机种子（保证可复现）
SAMPLE_SEED = 42

# ==================== 评估配置 ====================
# 评估指标
METRICS = [
    "accuracy",      # 准确率
    "precision",     # 精确率
    "recall",        # 召回率
    "f1",            # F1分数
    "confusion_matrix",  # 混淆矩阵
]

# 一致性分析
COMPUTE_CONSISTENCY = True  # 是否计算统一分类器和独立分类器的一致性
COMPUTE_COHEN_KAPPA = True  # 是否计算Cohen's Kappa系数

# ==================== 重试配置 ====================
# API调用失败时的重试策略
MAX_RETRIES = 5          # 最大重试次数
MAX_RETRY_TIME = 120     # 最大重试时间（秒）

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== 高级配置 ====================
# 是否启用思考模式（部分模型支持）
ENABLE_THINKING = False

# 结果格式（message或json格式）
RESULT_FORMAT = "message"

# ==================== 常用模型预设 ====================
MODEL_PRESETS = {
    "qwen-turbo": {
        "model_id": "qwen-turbo",
        "temperature": 0.1,
        "max_tokens": 1500,
        "cost_effective": True
    },
    "qwen-plus": {
        "model_id": "qwen-plus",
        "temperature": 0.1,
        "max_tokens": 2000,
        "balanced": True
    },
    "qwen-max": {
        "model_id": "qwen-max",
        "temperature": 0.1,
        "max_tokens": 2000,
        "high_quality": True
    }
}


def load_preset(preset_name: str) -> dict:
    """
    加载模型预设配置

    Args:
        preset_name: 预设名称 (qwen-turbo, qwen-plus, qwen-max)

    Returns:
        预设配置字典
    """
    return MODEL_PRESETS.get(preset_name, {})


def get_current_config() -> dict:
    """获取当前配置的字典形式"""
    return {
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "concurrency": CONCURRENCY,
        "chunk_size": CHUNK_SIZE,
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
    }


if __name__ == "__main__":
    # 打印当前配置
    print("="*60)
    print("当前LLM分类实验配置")
    print("="*60)
    for key, value in get_current_config().items():
        print(f"{key}: {value}")
    print("="*60)
