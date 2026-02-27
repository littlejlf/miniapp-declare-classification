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
            "model_id": MODEL_ID,
            "classifier": self.name,
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


# ==================== 评估函数 ====================
def load_ground_truth(sample_file: Path) -> Dict[str, Tuple[int, int]]:
    """加载真实标签"""
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ground_truth = {}
    for item in data:
        statement = item["statement"]
        label = item["label"]
        ground_truth[statement] = (label[0], label[1])
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
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
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


def print_evaluation_report(result_dir: Path, model_id: str):
    """打印评估报告"""
    sample_file = result_dir / f"sampled_200_list.json"
    if not sample_file.exists():
        print("\n未找到采样数据文件，跳过评估")
        return

    print("\n" + "="*70)
    print("开始评估实验结果...")
    print("="*70)

    # 加载真实标签
    ground_truth = load_ground_truth(sample_file)
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

    # 评估统一分类器
    print("\n" + "="*70)
    print("统一分类器评估")
    print("="*70)

    unified_nec = evaluate_classifier(ground_truth, unified_preds, "necessity")
    unified_amb = evaluate_classifier(ground_truth, unified_preds, "ambiguity")

    if "error" not in unified_nec:
        print(f"\n必要性维度:")
        print(f"  准确率: {unified_nec['accuracy']:.4f}")
        print(f"  精确率: {unified_nec['precision']:.4f}")
        print(f"  召回率: {unified_nec['recall']:.4f}")
        print(f"  F1分数: {unified_nec['f1']:.4f}")
        cm = unified_nec['confusion_matrix']
        print(f"  混淆矩阵: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    if "error" not in unified_amb:
        print(f"\n模糊性维度:")
        print(f"  准确率: {unified_amb['accuracy']:.4f}")
        print(f"  精确率: {unified_amb['precision']:.4f}")
        print(f"  召回率: {unified_amb['recall']:.4f}")
        print(f"  F1分数: {unified_amb['f1']:.4f}")
        cm = unified_amb['confusion_matrix']
        print(f"  混淆矩阵: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 评估独立分类器
    print("\n" + "="*70)
    print("独立分类器评估")
    print("="*70)

    nec_metrics = evaluate_classifier(ground_truth, nec_preds, "necessity")
    amb_metrics = evaluate_classifier(ground_truth, amb_preds, "ambiguity")

    if "error" not in nec_metrics:
        print(f"\n必要性分类器:")
        print(f"  准确率: {nec_metrics['accuracy']:.4f}")
        print(f"  精确率: {nec_metrics['precision']:.4f}")
        print(f"  召回率: {nec_metrics['recall']:.4f}")
        print(f"  F1分数: {nec_metrics['f1']:.4f}")

    if "error" not in amb_metrics:
        print(f"\n模糊性分类器:")
        print(f"  准确率: {amb_metrics['accuracy']:.4f}")
        print(f"  精确率: {amb_metrics['precision']:.4f}")
        print(f"  召回率: {amb_metrics['recall']:.4f}")
        print(f"  F1分数: {amb_metrics['f1']:.4f}")

    # 对比
    if "error" not in unified_nec and "error" not in nec_metrics:
        print(f"\n必要性对比:")
        print(f"  统一 vs 独立 F1: {unified_nec['f1']:.4f} vs {nec_metrics['f1']:.4f} (差异: {nec_metrics['f1']-unified_nec['f1']:.4f})")

    if "error" not in unified_amb and "error" not in amb_metrics:
        print(f"模糊性对比:")
        print(f"  统一 vs 独立 F1: {unified_amb['f1']:.4f} vs {amb_metrics['f1']:.4f} (差异: {amb_metrics['f1']-unified_amb['f1']:.4f})")

    # 保存报告
    report = {
        "model_id": model_id,
        "total_samples": len(ground_truth),
        "label_distribution": {
            "necessity_violations": sum(nec_labels),
            "necessity_total": len(nec_labels),
            "ambiguity_violations": sum(amb_labels),
            "ambiguity_total": len(amb_labels)
        },
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

    # 自动评估
    print_evaluation_report(OUTPUT_DIR, MODEL_ID)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断，结果已保存")
    except Exception as e:
        logger.error(f"失败: {e}")
        import traceback
        traceback.print_exc()
