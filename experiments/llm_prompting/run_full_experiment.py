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
from typing import List, Dict, Any, Optional, Deque, Tuple
from datetime import datetime
from collections import deque
import numpy as np

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
OUTPUT_DIR = project_root / "results" / "predictions" / MODEL_ID
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
        result = {
            "statement": statement,
            "model_id": MODEL_ID,
            "classifier": self.name,
            "result": "",
            "status": "pending",
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

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


# ==================== 评估函数 ====================
def load_ground_truth_full(input_file: Path) -> Dict[str, Tuple[int, int]]:
    """加载完整数据集的真实标签"""
    ground_truth = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")
                label = item.get("label", [])
                if statement and len(label) >= 2:
                    ground_truth[statement] = (label[0], label[1])
            except:
                continue
    return ground_truth


def load_predictions(result_file: Path) -> Dict[str, Dict]:
    """加载预测结果"""
    predictions = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")
                if item.get("status") == "success" and statement:
                    if statement not in predictions:
                        json_result = item.get("json")
                        if json_result:
                            predictions[statement] = json_result
                        else:
                            result_str = item.get("result", "")
                            if result_str:
                                try:
                                    json_result = json.loads(result_str)
                                    predictions[statement] = json_result
                                except:
                                    pass
            except:
                continue
    return predictions


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """计算分类指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }


def evaluate_classifier(
    ground_truth: Dict[str, Tuple[int, int]],
    predictions: Dict[str, Dict],
    dimension: str
) -> Dict:
    """评估单个分类器"""
    y_true, y_pred = [], []
    matched = 0

    for statement, (nec_true, amb_true) in ground_truth.items():
        true_label = nec_true if dimension == "necessity" else amb_true
        if statement in predictions:
            pred = predictions[statement]
            pred_label = int(pred.get(f"has_{dimension}_violation", False))
            y_true.append(true_label)
            y_pred.append(pred_label)
            matched += 1

    if not y_true:
        return {"error": "No valid predictions"}

    metrics = compute_metrics(y_true, y_pred)
    metrics["matched"] = matched
    metrics["total"] = len(ground_truth)
    return metrics


def print_evaluation_report_full(result_dir: Path, model_id: str, input_file: Path):
    """打印评估报告(完整数据集)"""
    print("\n" + "="*70)
    print("开始评估实验结果...")
    print("="*70)

    # 加载真实标签
    ground_truth = load_ground_truth_full(input_file)
    print(f"加载了 {len(ground_truth)} 条真实标签")

    # 统计标签分布
    nec_labels = [label[0] for label in ground_truth.values()]
    amb_labels = [label[1] for label in ground_truth.values()]

    print(f"\n标签分布:")
    print(f"  必要性违规: {sum(nec_labels)} / {len(nec_labels)} ({sum(nec_labels)/len(nec_labels)*100:.1f}%)")
    print(f"  模糊性违规: {sum(amb_labels)} / {len(amb_labels)} ({sum(amb_labels)/len(amb_labels)*100:.1f}%)")

    # 加载预测结果
    unified_preds = load_predictions(result_dir / "llm_unified_results.jsonl")
    nec_preds = load_predictions(result_dir / "llm_necessity_results.jsonl")
    amb_preds = load_predictions(result_dir / "llm_ambiguity_results.jsonl")

    print(f"\n预测结果:")
    print(f"  统一分类器: {len(unified_preds)} 条")
    print(f"  必要性分类器: {len(nec_preds)} 条")
    print(f"  模糊性分类器: {len(amb_preds)} 条")

    # 评估
    print("\n" + "="*70)
    print("统一分类器评估")
    print("="*70)

    unified_nec = evaluate_classifier(ground_truth, unified_preds, "necessity")
    unified_amb = evaluate_classifier(ground_truth, unified_preds, "ambiguity")

    if "error" not in unified_nec:
        print(f"\n必要性维度: Acc={unified_nec['accuracy']:.4f}, P={unified_nec['precision']:.4f}, R={unified_nec['recall']:.4f}, F1={unified_nec['f1']:.4f}")
    if "error" not in unified_amb:
        print(f"模糊性维度: Acc={unified_amb['accuracy']:.4f}, P={unified_amb['precision']:.4f}, R={unified_amb['recall']:.4f}, F1={unified_amb['f1']:.4f}")

    print("\n" + "="*70)
    print("独立分类器评估")
    print("="*70)

    nec_metrics = evaluate_classifier(ground_truth, nec_preds, "necessity")
    amb_metrics = evaluate_classifier(ground_truth, amb_preds, "ambiguity")

    if "error" not in nec_metrics:
        print(f"\n必要性分类器: Acc={nec_metrics['accuracy']:.4f}, P={nec_metrics['precision']:.4f}, R={nec_metrics['recall']:.4f}, F1={nec_metrics['f1']:.4f}")
    if "error" not in amb_metrics:
        print(f"模糊性分类器: Acc={amb_metrics['accuracy']:.4f}, P={amb_metrics['precision']:.4f}, R={amb_metrics['recall']:.4f}, F1={amb_metrics['f1']:.4f}")

    # 保存报告
    report = {
        "model_id": model_id,
        "total_samples": len(ground_truth),
        "unified_classifier": {
            "necessity": unified_nec if "error" not in unified_nec else {},
            "ambiguity": unified_amb if "error" not in unified_amb else {}
        },
        "independent_classifiers": {
            "necessity": nec_metrics if "error" not in nec_metrics else {},
            "ambiguity": amb_metrics if "error" not in amb_metrics else {}
        }
    }

    report_file = result_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已保存: {report_file}")
    print("="*70)


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

    # 自动评估
    print_evaluation_report_full(OUTPUT_DIR, MODEL_ID, INPUT_FILE)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断，结果已保存")
    except Exception as e:
        logger.error(f"失败: {e}")
