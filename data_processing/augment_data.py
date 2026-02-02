# -*- coding: utf-8 -*-
"""
数据增强脚本

使用LLM对隐私声明进行数据增强，生成语义相同的变体。
"""

import os
import sys
import asyncio
import json
import logging
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.llm_api import LLMAPIClient
from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)

# --- 1. 全局配置 ---
FINETUNED_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")

# API Key (必须通过环境变量设置)
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# --- 2. 数据增强提示词 ---

AUGMENTATION_PROMPT = """你是一个专业的文本改写助手。你的任务是对隐私声明进行数据增强，生成3-5条语义相同但表达方式不同的句子。

改写策略包括：
1. **语序重排**：调整句子中分句的顺序
2. **同义替换**：用意思相近的词汇替换原文中的词汇
3. **形容词/副词增强**：添加轻微的修饰词

请确保：
- 保持原文的核心语义不变
- 语气和风格保持一致
- 符合隐私声明的专业表达方式

请严格按照以下JSON格式返回：
{
  "augmented_texts": [
    "改写文本1",
    "改写文本2",
    "改写文本3"
  ]
}

不要添加任何额外的解释性文字。
"""


# --- 3. 数据增强函数 ---

async def augment_statement(
    statement: str,
    client: LLMAPIClient,
    semaphore: asyncio.Semaphore
) -> dict:
    """
    对单条声明进行数据增强

    Args:
        statement: 原始声明
        client: LLM客户端
        semaphore: 并发控制信号量

    Returns:
        增强结果字典
    """
    messages = [
        {"role": "system", "content": AUGMENTATION_PROMPT},
        {"role": "user", "content": f"请对以下隐私声明生成3-5条语义相同的改写：\n\n{statement}"}
    ]

    result = await client.call_api_async(messages, semaphore)

    # 解析增强文本
    augmented_texts = []
    if result.get("json") and "augmented_texts" in result["json"]:
        augmented_texts = result["json"]["augmented_texts"]
    elif result.get("content"):
        # 如果解析失败，尝试提取内容
        augmented_texts = [result["content"]]

    return {
        "original": statement,
        "augmented": augmented_texts,
        "count": len(augmented_texts),
        "status": result.get("status", "unknown")
    }


async def augment_from_file(
    input_file: Path,
    output_file: Path,
    model_id: str = FINETUNED_MODEL_ID,
    concurrency: int = 6
):
    """
    从文件读取声明并进行数据增强

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        model_id: 模型ID
        concurrency: 并发数
    """
    logger.info(f"读取输入文件: {input_file}")

    # 读取数据
    statements = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict):
                        statement = data.get("declare") or data.get("statement") or data.get("text")
                        if statement:
                            statements.append(statement)
                    else:
                        statements.append(str(data))
                except json.JSONDecodeError:
                    statements.append(line.strip())

    logger.info(f"共读取 {len(statements)} 条声明，准备进行数据增强")

    # 创建LLM客户端
    client = LLMAPIClient(
        model_id=model_id,
        api_key=API_KEY,
        temperature=1.2,  # 提高创造性
        max_tokens=2000
    )

    # 并发处理
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for stmt in statements:
        task = asyncio.create_task(augment_statement(stmt, client, semaphore))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 写入结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    total_augmented = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"处理失败: {res}")
                continue

            if res.get("status") == "success":
                success_count += 1
                total_augmented += res["count"]

                # 写入增强数据
                for aug_text in res["augmented"]:
                    output_data = {
                        "original": res["original"],
                        "augmented": aug_text,
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    logger.info(f"数据增强完成!")
    logger.info(f"成功处理: {success_count}/{len(statements)}")
    logger.info(f"生成增强样本: {total_augmented} 条")
    logger.info(f"结果已保存到: {output_file}")


# --- 4. 示例运行 ---

async def run_example():
    """示例运行函数"""

    # 数据文件路径
    input_file = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"

    # 输出文件（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "data" / "augmented" / f"augmented_{timestamp}.jsonl"

    # 执行数据增强
    await augment_from_file(
        input_file=input_file,
        output_file=output_file,
        model_id=FINETUNED_MODEL_ID,
        concurrency=6
    )


if __name__ == "__main__":
    # Windows兼容性设置
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行主函数
    asyncio.run(run_example())
