# -*- coding: utf-8 -*-
"""
LLM分类器测试脚本

特点：
1. 处理少量数据（默认10条）
2. 断点续传 - 已处理的自动跳过
3. 实时保存结果
4. 详细的错误日志
5. 进度显示

使用方法:
    python test_classifiers.py
"""

import os
import sys
import json
import time
import asyncio
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
# API配置 - 直接写在这里
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 模型配置
MODEL_ID = "qwen-plus"
TEMPERATURE = 0.1

# 测试配置
TEST_SIZE = 10  # 处理前10条数据
CONCURRENCY = 3  # 并发数（测试时用小一点的）
TIMEOUT = 60  # 单个请求超时时间（秒）

# 输出配置
OUTPUT_DIR = project_root / "results" / "test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输入文件
INPUT_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"

# ==================== 导入 ====================
from utils.llm_api import LLMAPIClient, APIStatus
from utils.logger import get_logger

logger = get_logger(__name__)


# ==================== 工具函数 ====================
def load_test_data(size: int = TEST_SIZE) -> List[Dict]:
    """加载���试数据"""
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= size:
                break
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"解析第{idx}行失败: {e}")

    logger.info(f"加载了 {len(data)} 条测试数据")
    return data


def get_processed_statements(output_file: Path) -> set:
    """获取已处理的statement集合"""
    processed = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    stmt = item.get("statement")
                    if stmt:
                        processed.add(stmt)
                except json.JSONDecodeError:
                    continue
    return processed


def save_result(result: Dict, output_file: Path):
    """实时保存单个结果"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def print_progress(current: int, total: int, success: int, failed: int):
    """打印进度"""
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r进度: [{bar}] {current}/{total} ({percent:.1f}%) | 成功: {success} | 失败: {failed}", end='', flush=True)


# ==================== 分类器测试 ====================
class ClassifierTester:
    """分类器测试器"""

    def __init__(self, name: str, prompt_file: Path, output_file: Path):
        self.name = name
        self.prompt_file = prompt_file
        self.output_file = output_file
        self.client = None
        self.system_prompt = None
        self.processed = set()

        # 统计
        self.total = 0
        self.success = 0
        self.failed = 0
        self.start_time = None

    def load_prompt(self):
        """加载提示词"""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"提示词文件不存在: {self.prompt_file}")

        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 提取变量赋值后的字符串
            if '=' in content:
                content = content.split('=', 1)[1].strip()
                if content.startswith('"""') or content.startswith("'''"):
                    content = content[3:]
                if content.endswith('"""') or content.endswith("'''"):
                    content = content[:-3]
            self.system_prompt = content.strip()

        logger.info(f"[{self.name}] 提示词已加载")

    def init_client(self):
        """初始化API客户端"""
        self.client = LLMAPIClient(
            model_id=MODEL_ID,
            api_key=DASHSCOPE_API_KEY,
            temperature=TEMPERATURE
        )
        logger.info(f"[{self.name}] API客户端已初始化")

    def load_processed(self):
        """加载已处理的数据"""
        self.processed = get_processed_statements(self.output_file)
        logger.info(f"[{self.name}] 已处理 {len(self.processed)} 条数据")

    async def classify_single(self, statement: str, semaphore: asyncio.Semaphore) -> Dict:
        """分类单条声明"""
        result = {
            "statement": statement,
            "result": "",
            "status": "pending",
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        try:
            api_result = await self.client.classify_statement(
                statement=statement,
                system_prompt=self.system_prompt,
                semaphore=semaphore
            )

            if api_result.get("status") == APIStatus.SUCCESS.value:
                result["result"] = api_result.get("content", "")
                result["status"] = "success"
                result["json"] = api_result.get("json")
            else:
                result["result"] = api_result.get("content", "")
                result["status"] = "format_error"
                result["error"] = "JSON解析失败"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    async def run_test(self, test_data: List[Dict]):
        """运行测试"""
        print(f"\n{'='*70}")
        print(f"测试: {self.name}")
        print(f"{'='*70}\n")

        self.load_prompt()
        self.init_client()
        self.load_processed()

        # 过滤未处理的数据
        to_process = []
        for item in test_data:
            stmt = item.get("statement")
            if stmt and stmt not in self.processed:
                to_process.append(item)

        self.total = len(to_process)
        logger.info(f"[{self.name}] 需要处理 {self.total} 条数据")

        if self.total == 0:
            print(f"[{self.name}] 所有数据已处理，跳过")
            return

        self.start_time = time.time()
        semaphore = asyncio.Semaphore(CONCURRENCY)

        # 创建任务
        tasks = []
        for item in to_process:
            stmt = item.get("statement")
            task = asyncio.create_task(self.classify_single(stmt, semaphore))
            tasks.append((task, item, stmt))

        # 执行任务
        for idx, (task, item, stmt) in enumerate(tasks):
            result = await task

            # 实时保存
            save_result(result, self.output_file)

            # 更新统计
            if result["status"] in ["success", "format_error"]:
                self.success += 1
            else:
                self.failed += 1

            # 打印进度
            print_progress(idx + 1, self.total, self.success, self.failed)

            # 打印详细结果
            if result["status"] == "success":
                print(f"\n✓ [{idx+1}] 成功")
            elif result["status"] == "format_error":
                print(f"\n⚠ [{idx+1}] JSON解析失败")
            else:
                print(f"\n✗ [{idx+1}] 失败: {result.get('error', 'Unknown')}")

        print(f"\n")

        # 打印总结
        elapsed = time.time() - self.start_time
        print(f"[{self.name}] 完成!")
        print(f"  总数: {self.total}")
        print(f"  成功: {self.success}")
        print(f"  失败: {self.failed}")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  平均: {elapsed/self.total:.2f}秒/条")
        print(f"  结果文件: {self.output_file}")


# ==================== 主程序 ====================
async def main():
    """主函数"""
    print("="*70)
    print("LLM分类器测试")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"数据: {INPUT_FILE}")
    print(f"测试数量: {TEST_SIZE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*70)

    # 加载测试数据
    test_data = load_test_data(TEST_SIZE)

    if not test_data:
        print("错误: 没有测试数据")
        return

    print(f"\n测试数据预览:")
    for idx, item in enumerate(test_data[:3], 1):
        stmt = item.get("statement", "")[:50]
        label = item.get("label", [])
        print(f"  {idx}. {stmt}... [label: {label}]")
    print()

    # 设置事件循环策略（Windows兼容）
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 定义测试器
    testers = [
        ClassifierTester(
            name="统一分类器",
            prompt_file=project_root / "prompts" / "classification_prompt.md",
            output_file=OUTPUT_DIR / "test_unified_results.jsonl"
        ),
        ClassifierTester(
            name="必要性分类器",
            prompt_file=project_root / "prompts" / "necessity_violation_prompt.md",
            output_file=OUTPUT_DIR / "test_necessity_results.jsonl"
        ),
        ClassifierTester(
            name="表述模糊分类器",
            prompt_file=project_root / "prompts" / "ambiguity_violation_prompt.md",
            output_file=OUTPUT_DIR / "test_ambiguity_results.jsonl"
        ),
    ]

    # 运行测试
    total_start = time.time()

    for tester in testers:
        try:
            await tester.run_test(test_data)
        except Exception as e:
            logger.error(f"{tester.name} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # 打印总结
    print(f"\n{'='*70}")
    print("所有测试完成!")
    print(f"{'='*70}")
    print(f"总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    print(f"结果目录: {OUTPUT_DIR}")
    print()
    print("生成的文件:")
    for file in OUTPUT_DIR.iterdir():
        size = file.stat().st_size
        print(f"  - {file.name} ({size} bytes)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        print("已处理的结果已保存")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
