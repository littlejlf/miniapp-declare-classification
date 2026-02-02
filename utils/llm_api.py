# -*- coding: utf-8 -*-
"""
LLM API 调用工具模块

提供统一的异步 LLM API 调用接口，支持重试、并发控制等功能。
"""

import os
import asyncio
import json
import logging
import platform
from typing import Dict, List, Optional, Any
from enum import Enum

import backoff
import dashscope
from dashscope.aigc.generation import AioGeneration
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse


class APIStatus(Enum):
    """API调用状态枚举"""
    SUCCESS = "success"
    FAILURE = "failure"
    FORMAT_ERROR = "format_error"


class LLMAPIClient:
    """LLM API 客户端封装类"""

    def __init__(
        self,
        model_id: str = "qwen-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.9
    ):
        """
        初始化 LLM API 客户端

        Args:
            model_id: 模型ID
            api_key: API密钥，如果为None则从环境变量读取
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p采样参数
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # 设置API Key
        if api_key:
            os.environ['DASHSCOPE_API_KEY'] = api_key
        elif 'DASHSCOPE_API_KEY' not in os.environ:
            raise ValueError("API Key must be provided or set in DASHSCOPE_API_KEY environment variable")

        # Windows兼容性设置
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        self.logger = logging.getLogger(__name__)

    def _should_give_up(self, e: Exception) -> bool:
        """
        判断是否应该放弃重试

        Args:
            e: 异常对象

        Returns:
            是否放弃重试
        """
        # 可以根据具体的异常类型来判断是否重试
        return False

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        max_time=120,
        giveup=lambda e: False,
        on_giveup=lambda details: logging.error("调用API失败次数过多或遇到不可重试错误，已放弃。")
    )
    async def call_api_async(
        self,
        messages: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
        result_format: str = 'message',
        enable_thinking: bool = False
    ) -> Dict[str, Any]:
        """
        异步调用 LLM API（带重试机制）

        Args:
            messages: 消息列表
            semaphore: 并发控制信号量
            result_format: 结果格式
            enable_thinking: 是否启用思考模式

        Returns:
            包含响应内容和状态的字典
        """
        async with semaphore:
            try:
                response: DashScopeAPIResponse = await AioGeneration.call(
                    model=self.model_id,
                    messages=messages,
                    result_format=result_format,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    enable_thinking=enable_thinking
                )

                if response.status_code == 200:
                    content = response.output.choices[0].message.get("content", "").strip()

                    # 尝试解析JSON
                    try:
                        json_obj = json.loads(content)
                        return {
                            "content": content,
                            "json": json_obj,
                            "status": APIStatus.SUCCESS.value
                        }
                    except json.JSONDecodeError:
                        self.logger.warning(f"返回的不是有效JSON: {content[:100]}...")
                        return {
                            "content": content,
                            "json": None,
                            "status": APIStatus.FORMAT_ERROR.value
                        }
                else:
                    error_msg = getattr(response, 'message', 'Unknown error')
                    self.logger.error(f"API调用返回非200状态码: {error_msg}")
                    raise dashscope.common.error.DashScopeAPIError(error_msg)

            except Exception as e:
                self.logger.error(f"API调用异常: {e}")
                raise

    async def classify_statement(
        self,
        statement: str,
        system_prompt: str,
        semaphore: asyncio.Semaphore,
        output_json: bool = True
    ) -> Dict[str, Any]:
        """
        对单个声明进行分类

        Args:
            statement: 待分类的声明文本
            system_prompt: 系统提示词
            semaphore: 并发控制信号量
            output_json: 是否期望JSON格式输出

        Returns:
            分类结果字典
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"目的声明：'{statement}'"}
        ]

        result = await self.call_api_async(messages, semaphore)

        # 添加原始声明到结果
        result["statement"] = statement

        return result

    async def batch_classify(
        self,
        statements: List[str],
        system_prompt: str,
        output_path: str,
        concurrency: int = 10,
        chunk_size: Optional[int] = None
    ) -> Dict[str, int]:
        """
        批量分类声明

        Args:
            statements: 声明列表
            system_prompt: 系统提示词
            output_path: 输出文件路径
            concurrency: 最大并发数
            chunk_size: 分块大小，如果为None则不分组

        Returns:
            统计信息字典
        """
        self.logger.info(f"开始处理 {len(statements)} 条声明，并发数: {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)
        tasks = []

        # 如果指定了分块大小，则分组处理
        if chunk_size:
            chunks = [statements[i:i + chunk_size] for i in range(0, len(statements), chunk_size)]
            self.logger.info(f"分成 {len(chunks)} 组进行处理")

            for idx, chunk in enumerate(chunks):
                self.logger.info(f"处理第 {idx + 1} 组，共 {len(chunk)} 条")

                for stmt in chunk:
                    task = asyncio.create_task(
                        self.classify_statement(stmt, system_prompt, semaphore)
                    )
                    tasks.append(task)

                # 处理当前组
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 写入结果
                success_count = self._write_results(results, output_path)

                self.logger.info(f"第 {idx + 1} 组完成，成功: {success_count}/{len(chunk)}")

                # 清空任务列表，准备下一组
                tasks = []
        else:
            # 不分组，一次性处理所有
            for stmt in statements:
                task = asyncio.create_task(
                    self.classify_statement(stmt, system_prompt, semaphore)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = self._write_results(results, output_path)

        return {
            "total": len(statements),
            "successful": success_count,
            "failed": len(statements) - success_count
        }

    def _write_results(self, results: List[Any], output_path: str) -> int:
        """
        将结果写入文件

        Args:
            results: 结果列表
            output_path: 输出路径

        Returns:
            成功写入的数量
        """
        success_count = 0

        with open(output_path, 'a', encoding='utf-8') as f:
            for res in results:
                if isinstance(res, Exception):
                    self.logger.error(f"任务失败: {res}")
                elif res and res.get('status') in [APIStatus.SUCCESS.value, APIStatus.FORMAT_ERROR.value]:
                    # 构建输出JSON
                    output = {
                        "statement": res.get("statement"),
                        "result": res.get("content"),
                        "status": res.get("status")
                    }
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    success_count += 1
                else:
                    self.logger.error(f"未知结果格式: {res}")

        return success_count


# 便捷函数
async def classify_with_llm(
    statements: List[str],
    system_prompt: str,
    model_id: str = "qwen-plus",
    output_path: str = "results.jsonl",
    api_key: Optional[str] = None,
    concurrency: int = 10
) -> Dict[str, int]:
    """
    使用LLM对声明进行分类的便捷函数

    Args:
        statements: 声明列表
        system_prompt: 系统提示词
        model_id: 模型ID
        output_path: 输出路径
        api_key: API密钥
        concurrency: 并发数

    Returns:
        统计信息
    """
    client = LLMAPIClient(model_id=model_id, api_key=api_key)
    return await client.batch_classify(
        statements=statements,
        system_prompt=system_prompt,
        output_path=output_path,
        concurrency=concurrency
    )
