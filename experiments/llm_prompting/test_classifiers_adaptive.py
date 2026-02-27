# -*- coding: utf-8 -*-
"""
LLM分类器测试脚本 - 自适应并发版本

特点：
1. 自适应并发控制 - 根据API响应时间动态调整并发数
2. 断点续传 - 已处理的自动跳过
3. 实时保存结果
4. 详细的错误日志
5. 进度显示

使用方法:
    python test_classifiers_adaptive.py
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
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
# API配置 - 直接写在这里
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 模型配置
MODEL_ID = "qwen3-14b"  # 或者 qwen-plus / qwen-max
TEMPERATURE = 0.1

# 测试配置
TEST_SIZE = 10  # 处理前10条数据（测试用）
SAMPLE_SIZE = 200  # 随机采样200条数据
SAMPLE_SEED = 42  # 随机种子（可复现）

# 并发配置 - 自适应
MIN_CONCURRENCY = 3         # 最小并发数
MAX_CONCURRENCY = 20        # 最大并发数
INITIAL_CONCURRENCY = 5     # 初始并发数

# 自适应控制参数
TARGET_RESPONSE_TIME = 3.0  # 目标响应时间（秒）
RESPONSE_TIME_WINDOW = 5    # 计算平均响应时间的窗口大小
ADJUSTMENT_THRESHOLD = 0.5  # 调整阈值（秒）

# 限流检测
RATE_LIMIT_DELAY = 5.0      # 触发限流后的延迟（秒）
MAX_RATE_LIMITS = 3         # 最大限流次数

# 输出配置
OUTPUT_DIR = project_root / "results" / "test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输入文件
INPUT_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"

# ==================== 导入 ====================
from utils.llm_api import LLMAPIClient, APIStatus
from utils.logger import get_logger

logger = get_logger(__name__)


# ==================== 自适应并发控制器 ====================
class AdaptiveConcurrencyController:
    """自适应并发控制器"""

    def __init__(
        self,
        min_concurrency: int = MIN_CONCURRENCY,
        max_concurrency: int = MAX_CONCURRENCY,
        initial_concurrency: int = INITIAL_CONCURRENCY,
        target_response_time: float = TARGET_RESPONSE_TIME,
        window_size: int = RESPONSE_TIME_WINDOW,
        adjustment_threshold: float = ADJUSTMENT_THRESHOLD
    ):
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.current_concurrency = initial_concurrency
        self.target_response_time = target_response_time
        self.window_size = window_size
        self.adjustment_threshold = adjustment_threshold

        # 响应时间追踪
        self.response_times = deque(maxlen=window_size)
        self.success_count = 0
        self.failure_count = 0
        self.rate_limit_count = 0

        # 统计
        self.total_requests = 0
        self.total_time = 0

    def record_response(self, response_time: float, success: bool, is_rate_limit: bool = False):
        """记录响应时间"""
        self.response_times.append(response_time)
        self.total_requests += 1
        self.total_time += response_time

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        if is_rate_limit:
            self.rate_limit_count += 1

    def get_average_response_time(self) -> Optional[float]:
        """获取平均响应时间"""
        if not self.response_times:
            return None
        return sum(self.response_times) / len(self.response_times)

    def get_current_concurrency(self) -> int:
        """获取当前并发数"""
        return self.current_concurrency

    def should_adjust(self) -> bool:
        """判断是否需要调整并发数"""
        if len(self.response_times) < self.window_size:
            return False

        avg_time = self.get_average_response_time()
        if avg_time is None:
            return False

        # 如果平均响应时间显著高于目标，降低并发
        if avg_time > self.target_response_time + self.adjustment_threshold:
            return True

        # 如果平均响应时间显著低于目标，可以提高并发
        if avg_time < self.target_response_time - self.adjustment_threshold:
            return True

        return False

    def adjust_concurrency(self) -> int:
        """调整并发数"""
        if len(self.response_times) < self.window_size:
            return self.current_concurrency

        avg_time = self.get_average_response_time()
        if avg_time is None:
            return self.current_concurrency

        old_concurrency = self.current_concurrency

        # 响应时间太长，降低并发
        if avg_time > self.target_response_time + self.adjustment_threshold:
            self.current_concurrency = max(
                self.min_concurrency,
                int(self.current_concurrency * 0.8)  # 降低20%
            )
            logger.info(f"响应时间过长 ({avg_time:.2f}s)，降低并发: {old_concurrency} -> {self.current_concurrency}")

        # 响应时间很短，可以提高并发
        elif avg_time < self.target_response_time - self.adjustment_threshold:
            self.current_concurrency = min(
                self.max_concurrency,
                int(self.current_concurrency * 1.2) + 1  # 提高20%并至少+1
            )
            logger.info(f"响应时间较短 ({avg_time:.2f}s)，提高并发: {old_concurrency} -> {self.current_concurrency}")

        return self.current_concurrency

    def handle_rate_limit(self) -> bool:
        """处理限流"""
        self.rate_limit_count += 1

        if self.rate_limit_count > MAX_RATE_LIMITS:
            logger.error(f"触发限流次数过多 ({self.rate_limit_count})，停止处理")
            return False

        # 大幅降低并发
        old_concurrency = self.current_concurrency
        self.current_concurrency = max(
            self.min_concurrency,
            int(self.current_concurrency * 0.5)  # 降低50%
        )
        logger.warning(
            f"触发API限流，降低并发并等待: "
            f"{old_concurrency} -> {self.current_concurrency}, "
            f"等待 {RATE_LIMIT_DELAY} 秒"
        )

        time.sleep(RATE_LIMIT_DELAY)
        return True

    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_time = self.get_average_response_time()
        return {
            "current_concurrency": self.current_concurrency,
            "average_response_time": avg_time,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "rate_limit_count": self.rate_limit_count,
            "total_requests": self.total_requests,
            "success_rate": self.success_count / self.total_requests if self.total_requests > 0 else 0
        }


# ==================== 工具函数 ====================
def load_test_data(size: int = TEST_SIZE) -> List[Dict]:
    """加载测试数据"""
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


def print_progress(current: int, total: int, success: int, failed: int, controller: AdaptiveConcurrencyController):
    """打印进度"""
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    stats = controller.get_stats()
    concurrency = stats['current_concurrency']
    avg_time = stats.get('average_response_time', 0)

    print(
        f"\r进度: [{bar}] {current}/{total} ({percent:.1f}%) | "
        f"成功: {success} | 失败: {failed} | "
        f"并发: {concurrency} | 平均响应: {avg_time:.2f}s",
        end='', flush=True
    )


# ==================== 分类器测试 ====================
class AdaptiveClassifierTester:
    """带自适应并发的分类器测试器"""

    def __init__(self, name: str, prompt_file: Path, output_file: Path):
        self.name = name
        self.prompt_file = prompt_file
        self.output_file = output_file
        self.client = None
        self.system_prompt = None
        self.processed = set()
        self.controller = AdaptiveConcurrencyController()

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

    async def classify_single(self, statement: str) -> Dict:
        """分类单条声明"""
        start_time = time.time()
        result = {
            "statement": statement,
            "result": "",
            "status": "pending",
            "error": None,
            "response_time": 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            semaphore = asyncio.Semaphore(1)  # 单个任务的信号量
            api_result = await self.client.classify_statement(
                statement=statement,
                system_prompt=self.system_prompt,
                semaphore=semaphore
            )

            response_time = time.time() - start_time
            result["response_time"] = response_time

            # 记录响应时间
            is_rate_limit = "rate limit" in str(api_result.get("status", "")).lower()
            success = api_result.get("status") == APIStatus.SUCCESS.value

            self.controller.record_response(response_time, success, is_rate_limit)

            if success:
                result["result"] = api_result.get("content", "")
                result["status"] = "success"
                result["json"] = api_result.get("json")
            elif "format" in str(api_result.get("status", "")):
                result["result"] = api_result.get("content", "")
                result["status"] = "format_error"
                result["error"] = "JSON解析失败"
            else:
                result["status"] = "failed"
                result["error"] = api_result.get("status", "Unknown")

        except Exception as e:
            response_time = time.time() - start_time
            result["response_time"] = response_time
            result["status"] = "failed"
            result["error"] = str(e)

            # 检查是否是限流
            error_str = str(e).lower()
            is_rate_limit = (
                "rate limit" in error_str or
                "429" in error_str or
                "too many requests" in error_str
            )

            self.controller.record_response(response_time, False, is_rate_limit)

            if is_rate_limit:
                can_continue = self.controller.handle_rate_limit()
                if not can_continue:
                    raise RuntimeError("触发限流次数过多")

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
        idx = 0

        while idx < self.total:
            # 获取当前并发数
            concurrency = self.controller.get_current_concurrency()

            # 创建一批任务
            batch_size = min(concurrency, self.total - idx)
            batch = to_process[idx:idx + batch_size]

            # 执行批次
            tasks = [self.classify_single(item.get("statement")) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"任务异常: {result}")
                    self.failed += 1
                    continue

                # 实时保存
                save_result(result, self.output_file)

                # 更新统计
                if result["status"] in ["success", "format_error"]:
                    self.success += 1
                else:
                    self.failed += 1

                idx += 1

                # 打印进度
                print_progress(idx, self.total, self.success, self.failed, self.controller)

            # 检查是否需要调整并发
            if self.controller.should_adjust():
                self.controller.adjust_concurrency()

        print(f"\n")

        # 打印总结
        elapsed = time.time() - self.start_time
        stats = self.controller.get_stats()

        print(f"[{self.name}] 完成!")
        print(f"  总数: {self.total}")
        print(f"  成功: {self.success}")
        print(f"  失败: {self.failed}")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  平均: {elapsed/self.total:.2f}秒/条")
        print(f"  最终并发数: {stats['current_concurrency']}")
        print(f"  平均响应时间: {stats.get('average_response_time', 0):.2f}秒")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  结果文件: {self.output_file}")


# ==================== 主程序 ====================
async def main():
    """主函数"""
    print("="*70)
    print("LLM分类器测试 (自适应并发版本)")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"数据: {INPUT_FILE}")
    print(f"测试数量: {TEST_SIZE}")
    print(f"并发范围: {MIN_CONCURRENCY}-{MAX_CONCURRENCY}")
    print(f"初始并发: {INITIAL_CONCURRENCY}")
    print(f"目标响应时间: {TARGET_RESPONSE_TIME}秒")
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
        AdaptiveClassifierTester(
            name="统一分类器",
            prompt_file=project_root / "prompts" / "classification_prompt.md",
            output_file=OUTPUT_DIR / "test_unified_results.jsonl"
        ),
        AdaptiveClassifierTester(
            name="必要性分类器",
            prompt_file=project_root / "prompts" / "necessity_violation_prompt.md",
            output_file=OUTPUT_DIR / "test_necessity_results.jsonl"
        ),
        AdaptiveClassifierTester(
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
