# -*- coding: utf-8 -*-
"""
LLM分类实验 - 采样200条版本

特点：
1. 随机采样200条数据进行测试
2. 自适应并发控制
3. 断点续传
4. 实时保存结果
5. 完整评估报告

使用方法:
    python run_sampled_experiment.py
"""

import os
import sys
import json
import time
import asyncio
import platform
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Deque
from datetime import datetime
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
# API配置
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 模型配置
MODEL_ID = "qwen3-14b"  # 使用qwen3-14b模型
TEMPERATURE = 0.1

# 采样配置
SAMPLE_SIZE = 200  # 采样数量
SAMPLE_SEED = 42   # 随机种子

# 并发配置 - 自适应
MIN_CONCURRENCY = 3         # 最小并发数
MAX_CONCURRENCY = 15        # 最大并发数
INITIAL_CONCURRENCY = 5     # 初始并发数

# 自适应控制参数
TARGET_RESPONSE_TIME = 2.5
RESPONSE_TIME_WINDOW = 10
ADJUSTMENT_THRESHOLD = 0.5
RATE_LIMIT_DELAY = 5.0
MAX_RATE_LIMITS = 5

# 输出配置
OUTPUT_DIR = project_root / "results" / "predictions"
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
        window_size: int = RESPONSE_TIME_WINDOW
    ):
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.current_concurrency = initial_concurrency
        self.target_response_time = target_response_time
        self.window_size = window_size

        self.response_times: Deque[float] = deque(maxlen=window_size)
        self.success_count = 0
        self.failure_count = 0
        self.rate_limit_count = 0

    def record_response(self, response_time: float, success: bool, is_rate_limit: bool = False):
        """记录响应"""
        self.response_times.append(response_time)
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

    def should_adjust(self) -> bool:
        """判断是否需要调整"""
        if len(self.response_times) < self.window_size // 2:
            return False
        avg_time = self.get_average_response_time()
        return avg_time is not None and abs(avg_time - self.target_response_time) > 0.5

    def adjust_concurrency(self) -> int:
        """调整并发数"""
        if len(self.response_times) < self.window_size // 2:
            return self.current_concurrency

        avg_time = self.get_average_response_time()
        if avg_time is None:
            return self.current_concurrency

        old = self.current_concurrency

        if avg_time > self.target_response_time + 0.5:
            self.current_concurrency = max(self.min_concurrency, int(self.current_concurrency * 0.85))
        elif avg_time < self.target_response_time - 0.5:
            self.current_concurrency = min(self.max_concurrency, int(self.current_concurrency * 1.15) + 1)

        if old != self.current_concurrency:
            logger.info(f"并发调整: {old} -> {self.current_concurrency} (响应时间: {avg_time:.2f}s)")

        return self.current_concurrency

    def handle_rate_limit(self) -> bool:
        """处理限流"""
        self.rate_limit_count += 1
        if self.rate_limit_count > MAX_RATE_LIMITS:
            return False

        old = self.current_concurrency
        self.current_concurrency = max(self.min_concurrency, int(self.current_concurrency * 0.5))
        logger.warning(f"触发限流，等待{RATE_LIMIT_DELAY}秒后降低并发: {old} -> {self.current_concurrency}")
        time.sleep(RATE_LIMIT_DELAY)
        return True

    def get_current_concurrency(self) -> int:
        """获取当前并发数"""
        return self.current_concurrency


# ==================== 采样函数 ====================
def load_all_data() -> List[Dict]:
    """加载所有数据"""
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def sample_data(data: List[Dict], sample_size: int, seed: int) -> List[Dict]:
    """随机采样数据"""
    random.seed(seed)
    sampled = random.sample(data, min(sample_size, len(data)))

    # 统计标签分布
    labels = {}
    for item in sampled:
        label = tuple(item.get("label", []))
        labels[label] = labels.get(label, 0) + 1

    logger.info(f"从 {len(data)} 条数据中采样 {len(sampled)} 条")
    logger.info(f"标签分布: {labels}")

    return sampled


def get_processed_statements(output_file: Path) -> set:
    """获取已处理的statement集合"""
    processed = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if stmt := item.get("statement"):
                        processed.add(stmt)
                except:
                    pass
    return processed


def save_result(result: Dict, output_file: Path):
    """实时保存单个结果"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


# ==================== 分类器 ====================
class AdaptiveClassifier:
    """带自适应并发的分类器"""

    def __init__(self, name: str, prompt_file: Path, output_file: Path):
        self.name = name
        self.prompt_file = prompt_file
        self.output_file = output_file
        self.client = None
        self.system_prompt = None
        self.processed = set()
        self.controller = AdaptiveConcurrencyController()

    def load_prompt(self):
        """加载提示词"""
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
        """初始化客户端"""
        self.client = LLMAPIClient(model_id=MODEL_ID, api_key=DASHSCOPE_API_KEY, temperature=TEMPERATURE)

    def load_processed(self):
        """加载已处理数据"""
        self.processed = get_processed_statements(self.output_file)
        logger.info(f"[{self.name}] 已处理 {len(self.processed)} 条")

    async def classify_single(self, statement: str) -> Dict:
        """分类单条"""
        start = time.time()
        result = {
            "statement": statement,
            "result": "",
            "status": "pending",
            "error": None,
            "response_time": 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            api_result = await self.client.classify_statement(
                statement, self.system_prompt, asyncio.Semaphore(1)
            )
            rt = time.time() - start
            result["response_time"] = rt

            is_rl = "rate limit" in str(api_result.get("status", "")).lower()
            success = api_result.get("status") == APIStatus.SUCCESS.value
            self.controller.record_response(rt, success, is_rl)

            if success:
                result.update({
                    "result": api_result.get("content", ""),
                    "status": "success",
                    "json": api_result.get("json")
                })
            elif "format" in str(api_result.get("status", "")):
                result.update({
                    "result": api_result.get("content", ""),
                    "status": "format_error",
                    "error": "JSON解析失败"
                })
            else:
                result["status"] = "failed"
                result["error"] = api_result.get("status")

        except Exception as e:
            rt = time.time() - start
            result.update({"response_time": rt, "status": "failed", "error": str(e)})
            is_rl = any(x in str(e).lower() for x in ["rate limit", "429", "too many"])
            self.controller.record_response(rt, False, is_rl)
            if is_rl and not self.controller.handle_rate_limit():
                raise RuntimeError("限流次数过多")

        return result

    async def run(self, data: List[Dict]):
        """运行分类"""
        print(f"\n{'='*70}\n{self.name}\n{'='*70}\n")
        self.load_prompt()
        self.init_client()
        self.load_processed()

        # 过滤未处理的数据
        to_process = [d for d in data if d.get("statement") not in self.processed]
        total = len(to_process)

        if total == 0:
            print("所有数据已处理")
            return

        print(f"需要处理: {total} 条")
        start_time = time.time()
        idx = success = failed = 0

        while idx < total:
            concurrency = self.controller.get_current_concurrency()
            batch = to_process[idx:idx + concurrency]

            results = await asyncio.gather(
                *[self.classify_single(d.get("statement")) for d in batch],
                return_exceptions=True
            )

            for r in results:
                if isinstance(r, Exception):
                    failed += 1
                    continue

                save_result(r, self.output_file)

                if r["status"] in ["success", "format_error"]:
                    success += 1
                else:
                    failed += 1
                idx += 1

                avg_rt = self.controller.get_average_response_time() or 0
                print(
                    f"\r进度: {idx}/{total} ({idx/total*100:.1f}%) | "
                    f"成功: {success} | 失败: {failed} | "
                    f"并发: {concurrency} | 平均响应: {avg_rt:.2f}s",
                    end='', flush=True
                )

            if self.controller.should_adjust():
                self.controller.adjust_concurrency()

        elapsed = time.time() - start_time
        print(f"\n\n完成! 总数: {total} | 成功: {success} | 失败: {failed} | 耗时: {elapsed:.1f}秒 | 平均: {elapsed/total:.2f}秒/条\n")


# ==================== 主程序 ====================
async def main():
    print("="*70)
    print("LLM分类实验 - 采样200条")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"并发范围: {MIN_CONCURRENCY}-{MAX_CONCURRENCY}")
    print(f"采样数量: {SAMPLE_SIZE}")
    print(f"随机种子: {SAMPLE_SEED}")
    print(f"数据源: {INPUT_FILE}")
    print("="*70)

    # 加载所有数据
    print("\n加载数据...")
    all_data = load_all_data()
    print(f"数据总量: {len(all_data)} 条")

    # 采样
    print(f"\n随机采样 {SAMPLE_SIZE} 条...")
    sampled_data = sample_data(all_data, SAMPLE_SIZE, SAMPLE_SEED)

    # 保存采样列表
    sample_list_file = OUTPUT_DIR / f"sampled_{SAMPLE_SIZE}_list.json"
    with open(sample_list_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    print(f"采样列表已保存: {sample_list_file}\n")

    # 定义分类器
    classifiers = [
        AdaptiveClassifier(
            "统一分类器",
            project_root / "prompts" / "classification_prompt.md",
            OUTPUT_DIR / "llm_unified_results.jsonl"
        ),
        AdaptiveClassifier(
            "必要性分类器",
            project_root / "prompts" / "necessity_violation_prompt.md",
            OUTPUT_DIR / "llm_necessity_results.jsonl"
        ),
        AdaptiveClassifier(
            "表述模糊分类器",
            project_root / "prompts" / "ambiguity_violation_prompt.md",
            OUTPUT_DIR / "llm_ambiguity_results.jsonl"
        ),
    ]

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    total_start = time.time()

    for c in classifiers:
        try:
            await c.run(sampled_data)
        except Exception as e:
            logger.error(f"{c.name} 失败: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    print(f"结果目录: {OUTPUT_DIR}")
    print("="*70)

    print("\n生成的文件:")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断，结果已保存")
    except Exception as e:
        logger.error(f"失败: {e}")
        import traceback
        traceback.print_exc()
