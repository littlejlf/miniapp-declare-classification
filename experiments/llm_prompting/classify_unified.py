# -*- coding: utf-8 -*-
"""
统一LLM分类器 - 同时评估必要性和表述模糊

使用原始的 classification_prompt.md，在一个提示词中同时评估两个维度。
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

from utils.llm_api import LLMAPIClient
from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)

# --- 全局配置 ---
FINETUNED_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


def load_system_prompt(prompt_file: Optional[str] = None) -> str:
    """加载系统提示词"""
    if prompt_file is None:
        prompt_file = project_root / "prompts" / "classification_prompt.md"

    if not prompt_file.exists():
        logger.warning(f"提示词文件不存在: {prompt_file}")
        return "你是一位隐私政策审计专家。"

    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 提取变量赋值后的字符串
        if '=' in content:
            content = content.split('=', 1)[1].strip()
            if content.startswith('"""') or content.startswith("'''"):
                content = content[3:]
            if content.endswith('"""') or content.endswith("'''"):
                content = content[:-3]
        return content.strip()


async def classify_from_file(
    input_file: Path,
    output_file: Path,
    model_id: str = FINETUNED_MODEL_ID,
    concurrency: int = 10,
    chunk_size: Optional[int] = None
):
    """从文件读取声明并进行统一分类"""
    logger.info(f"[统一分类器] 读取输入文件: {input_file}")

    statements = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict):
                        statement = data.get("declare") or data.get("statement") or data.get("text")
                    else:
                        statement = str(data)
                    if statement:
                        statements.append(statement)
                except json.JSONDecodeError:
                    statements.append(line.strip())

    logger.info(f"[统一分类器] 共读取 {len(statements)} 条声明")

    system_prompt = load_system_prompt()
    client = LLMAPIClient(model_id=model_id, api_key=API_KEY, temperature=0.1)

    stats = await client.batch_classify(
        statements=statements,
        system_prompt=system_prompt,
        output_path=str(output_file),
        concurrency=concurrency,
        chunk_size=chunk_size
    )

    logger.info(f"[统一分类器] 完成! 总数: {stats['total']}, 成功: {stats['successful']}, 失败: {stats['failed']}")
    logger.info(f"[统一分类器] 结果已保存到: {output_file}")
    return stats


async def run_example():
    """示例运行函数"""
    input_file = project_root / "data" / "raw" / "all_declares.jsonl"
    output_dir = project_root / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "llm_unified_results.jsonl"

    await classify_from_file(
        input_file=input_file,
        output_file=output_file,
        model_id=FINETUNED_MODEL_ID,
        concurrency=10,
        chunk_size=100
    )


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_example())
