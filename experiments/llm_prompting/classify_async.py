# -*- coding: utf-8 -*-
"""
LLM提示词分类实验脚本

使用大���言模型对隐私声明进行分类，支持异步并发调用。
"""

import os
import sys
import json
import asyncio
import platform
import logging
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.llm_api import LLMAPIClient, classify_with_llm
from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)

# --- 1. 全局配置 ---
# 模型ID (可以通过环境变量覆盖)
FINETUNED_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")

# API Key (必须通过环境变量设置)
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 系统提示词 - 从文件读取
def load_system_prompt(prompt_file: Optional[str] = None) -> str:
    """
    加载系统提示词

    Args:
        prompt_file: 提示词文件路径，如果为None则使用默认路径

    Returns:
        系统提示词字符串
    """
    if prompt_file is None:
        prompt_file = project_root / "prompts" / "classification_prompt.md"

    if not prompt_file.exists():
        logger.warning(f"提示词文件不存在: {prompt_file}，使用默认提示词")
        return "你是一位隐私政策审计专家。请分析用户提供的隐私声明，判断其必要性。"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


# --- 2. 主处理函数 ---

async def classify_from_file(
    input_file: Path,
    output_file: Path,
    model_id: str = FINETUNED_MODEL_ID,
    concurrency: int = 10,
    chunk_size: Optional[int] = None
):
    """
    从文件读取声明并进行分类

    Args:
        input_file: 输入文件路径 (JSONL格式)
        output_file: 输出文件路径
        model_id: 模型ID
        concurrency: 并发数
        chunk_size: 分块大小
    """
    logger.info(f"读取输入文件: {input_file}")

    # 读取数据
    statements = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    # 假设格式为 {"declare": "..."} 或直接是文本
                    if isinstance(data, dict):
                        statement = data.get("declare") or data.get("statement") or data.get("text")
                    else:
                        statement = str(data)

                    if statement:
                        statements.append(statement)
                except json.JSONDecodeError:
                    # 如果不是JSON，直接作为文本处理
                    statements.append(line.strip())

    logger.info(f"共读取 {len(statements)} 条声明")

    # 加载系统提示词
    system_prompt = load_system_prompt()

    # 创建LLM客户端
    client = LLMAPIClient(
        model_id=model_id,
        api_key=API_KEY,
        temperature=0.1
    )

    # 执行分类
    stats = await client.batch_classify(
        statements=statements,
        system_prompt=system_prompt,
        output_path=str(output_file),
        concurrency=concurrency,
        chunk_size=chunk_size
    )

    logger.info(f"分类完成! 总数: {stats['total']}, 成功: {stats['successful']}, 失败: {stats['failed']}")
    logger.info(f"结果已保存到: {output_file}")


async def run_example():
    """示例运行函数"""

    # 数据文件路径
    input_file = project_root / "data" / "raw" / "all_declares.jsonl"
    output_file = project_root / "results" / "predictions" / "llm_classification_results.jsonl"

    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 执行分类
    await classify_from_file(
        input_file=input_file,
        output_file=output_file,
        model_id=FINETUNED_MODEL_ID,
        concurrency=10,
        chunk_size=100  # 每100条为一组，便于追踪进度
    )


# --- 3. 入口函数 ---

if __name__ == "__main__":
    # Windows兼容性设置
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行主函数
    asyncio.run(run_example())
