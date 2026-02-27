# -*- coding: utf-8 -*-
"""
测试模型是否可用

用于验证不同的模型ID在阿里云DashScope API上是否可用
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# API配置
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 要测试的模型列表
MODELS_TO_TEST = [
    "qwen-3-14b",
    "qwen3-14b",
    "qwen-plus",
    "qwen-turbo",
    "qwen-max",
    "qwen-long",
]

# 测试提示词
TEST_PROMPT = "你是一个测试助手。请回复'测试成功'。"
TEST_MESSAGE = "你好"


async def test_model(model_id: str, api_key: str) -> dict:
    """测试单个模型是否可用"""
    try:
        import dashscope
        from dashscope import Generation

        # 设置API key
        os.environ["DASHSCOPE_API_KEY"] = api_key

        response = await asyncio.to_thread(
            Generation.call,
            model=model_id,
            prompt=TEST_MESSAGE,
            result_format='message'
        )

        if response.status_code == 200:
            return {
                "model": model_id,
                "status": "✓ 可用",
                "response": response.output.choices[0].message.content[:50]
            }
        else:
            return {
                "model": model_id,
                "status": f"✗ 不可用 ({response.status_code})",
                "error": response.message
            }

    except Exception as e:
        return {
            "model": model_id,
            "status": f"✗ 错误",
            "error": str(e)[:100]
        }


async def main():
    print("="*70)
    print("阿里云DashScope模型可用性测试")
    print("="*70)
    print(f"API Key: {DASHSCOPE_API_KEY[:20]}...")
    print(f"测试模型数量: {len(MODELS_TO_TEST)}")
    print("="*70)
    print()

    results = []

    for model_id in MODELS_TO_TEST:
        print(f"测试: {model_id}...", end=" ", flush=True)
        result = await test_model(model_id, DASHSCOPE_API_KEY)
        results.append(result)
        print(result["status"])

        if "response" in result:
            print(f"  └─ 回复: {result['response']}")

    print()
    print("="*70)
    print("测试结果汇总")
    print("="*70)
    print()

    for result in results:
        print(f"{result['model']:30} {result['status']}")

    print()
    print("="*70)
    print("可用的模型:")
    available = [r["model"] for r in results if "可用" in r["status"]]
    if available:
        for m in available:
            print(f"  - {m}")
    else:
        print("  无可用模型")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
