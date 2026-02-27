# -*- coding: utf-8 -*-
"""
LLM分类完整实验脚本 - 自适应并发版本

功能：
1. 自适应并发控制（根据API响应时间自动调整）
2. 处理全部1137条数据
3. 断点续传
4. 实时保存
5. 完整评估报告

使用方法:
    python run_full_experiment.py
"""

import os
import sys
import json
import time
import asyncio
import platform
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
MODEL_ID = "qwen-plus"
TEMPERATURE = 0.1

# 并发配置 - 自适应
MIN_CONCURRENCY = 5
MAX_CONCURRENCY = 15
INITIAL_CONCURRENCY = 8

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
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if stmt := item.get("statement"):
                            self.processed.add(stmt)
                    except:
                        pass
        logger.info(f"[{self.name}] 已处理 {len(self.processed)} 条")

    async def classify_single(self, statement: str) -> Dict:
        """分类单条"""
        start = time.time()
        result = {"statement": statement, "result": "", "status": "pending", "error": None, "timestamp": datetime.now().isoformat()}

        try:
            api_result = await self.client.classify_statement(statement, self.system_prompt, asyncio.Semaphore(1))
            rt = time.time() - start

            is_rl = "rate limit" in str(api_result.get("status", "")).lower()
            success = api_result.get("status") == APIStatus.SUCCESS.value
            self.controller.record_response(rt, success, is_rl)

            if success:
                result.update({"result": api_result.get("content", ""), "status": "success", "json": api_result.get("json")})
            elif "format" in str(api_result.get("status", "")):
                result.update({"result": api_result.get("content", ""), "status": "format_error", "error": "JSON解析失败"})
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

            results = await asyncio.gather(*[self.classify_single(d.get("statement")) for d in batch], return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    failed += 1
                    continue

                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

                if r["status"] in ["success", "format_error"]:
                    success += 1
                else:
                    failed += 1
                idx += 1

                stats = self.controller.get_stats() if hasattr(self.controller, 'get_stats') else {}
                avg_rt = self.controller.get_average_response_time() or 0
                print(f"\r进度: {idx}/{total} ({idx/total*100:.1f}%) | 成功: {success} | 失败: {failed} | 并发: {concurrency} | 平均响应: {avg_rt:.2f}s", end='', flush=True)

            if self.controller.should_adjust():
                self.controller.adjust_concurrency()

        elapsed = time.time() - start_time
        print(f"\n\n完成! 总数: {total} | 成功: {success} | 失败: {failed} | 耗时: {elapsed:.1f}秒 | 平均: {elapsed/total:.2f}秒/条\n")


# ==================== 主程序 ====================
async def main():
    print("="*70)
    print("LLM分类完整实验 (自适应并发)")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"并发范围: {MIN_CONCURRENCY}-{MAX_CONCURRENCY}")
    print(f"数据: {INPUT_FILE}")
    print("="*70)

    # 加载数据
    data = []
    with open(INPUT_FILE) as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                pass

    print(f"\n数据总量: {len(data)} 条\n")

    # 定义分类器
    classifiers = [
        AdaptiveClassifier("统一分类器", project_root / "prompts" / "classification_prompt.md", OUTPUT_DIR / "llm_unified_results.jsonl"),
        AdaptiveClassifier("必要性分类器", project_root / "prompts" / "necessity_violation_prompt.md", OUTPUT_DIR / "llm_necessity_results.jsonl"),
        AdaptiveClassifier("表述模糊分类器", project_root / "prompts" / "ambiguity_violation_prompt.md", OUTPUT_DIR / "llm_ambiguity_results.jsonl"),
    ]

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    total_start = time.time()

    for c in classifiers:
        try:
            await c.run(data)
        except Exception as e:
            logger.error(f"{c.name} 失败: {e}")

    print(f"\n{'='*70}")
    print(f"总耗时: {time.time()-total_start:.1f}秒 ({(time.time()-total_start)/60:.1f}分钟)")
    print(f"结果目录: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断，结果已保存")
    except Exception as e:
        logger.error(f"失败: {e}")
