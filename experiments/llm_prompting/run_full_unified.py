# -*- coding: utf-8 -*-
"""
LLM统一分类器 - 完整数据集运行

特点：
1. 使用 httpx.AsyncClient 连接池控制总并发
2. 使用 Tenacity 控制重试逻辑
3. 只运行统一分类器
4. 对所有1137条标注数据进行分类
5. 实时保存结果，支持断点续传
6. 生成完整评估报告

使用方法:
    python run_full_unified.py
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
# API配置
DASHSCOPE_API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"

# 模型配置
MODEL_ID = "qwen3-32b"
TEMPERATURE = 0.1
MAX_TOKENS = 2000

# 并发配置
MAX_CONCURRENT_REQUESTS = 10  # httpx连接池最大连接数
REQUEST_TIMEOUT = 60.0  # 请求超时时间

# 重试配置 (Tenacity)
MAX_RETRIES = 3  # 最大重试次数
RETRY_WAIT_MIN = 1.0  # 最小等待时间(秒)
RETRY_WAIT_MAX = 10.0  # 最大等待时间(秒)

# 输出配置
OUTPUT_DIR = project_root / "results" / "predictions" / MODEL_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输入文件
INPUT_FILE = project_root / "data" / "raw" / "aggregate_datas_label.jsonl"

# 统一分类器提示词
PROMPT_FILE = project_root / "prompts" / "classification_prompt.md"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== DashScope API 调用 ====================
class DashScopeClient:
    """DashScope API 客户端"""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def call_api(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """调用 DashScope API (带重试)"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_id,
            "input": {
                "messages": messages
            },
            "parameters": {
                "result_format": "message",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "enable_thinking": False
            }
        }

        response = await client.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        return response.json()


# ==================== 数据处理 ====================
def load_all_data() -> List[Dict]:
    """加载所有标注数据"""
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


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


def parse_json_from_response(content: str) -> Optional[Dict]:
    """从响应内容中解析JSON"""
    try:
        # 直接解析
        return json.loads(content)
    except:
        # 尝试去除 ```json 和 ``` 标记
        if '```json' in content:
            content = content.split('```json', 1)[1]
        if '```' in content:
            content = content.split('```')[0].strip()
        try:
            return json.loads(content)
        except:
            return None


# ==================== 主分类器 ====================
async def classify_statement(
    statement: str,
    system_prompt: str,
    api_client: DashScopeClient,
    client: httpx.AsyncClient
) -> Dict:
    """对单条声明进行分类"""

    start_time = time.time()
    result = {
        "statement": statement,
        "model_id": MODEL_ID,
        "classifier": "统一分类器",
        "result": "",
        "status": "pending",
        "error": None,
        "response_time": 0,
        "timestamp": datetime.now().isoformat()
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": statement}
    ]

    try:
        response = await api_client.call_api(client, messages)

        # DashScope 原生 API 响应格式
        if response.get("output") and response["output"].get("choices"):
            content = response["output"]["choices"][0]["message"]["content"]
            result["result"] = content

            # 尝试解析JSON
            parsed_json = parse_json_from_response(content)
            if parsed_json:
                result["json"] = parsed_json
                result["status"] = "success"
            else:
                result["status"] = "format_error"
                result["error"] = "JSON解析失败"
        else:
            result["status"] = "failed"
            result["error"] = "响应格式错误"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"分类失败: {e}")

    result["response_time"] = time.time() - start_time
    return result


async def process_batch(
    statements: List[str],
    system_prompt: str,
    api_client: DashScopeClient,
    output_file: Path
):
    """处理一个批次的数据"""

    # 创建httpx异步客户端，带连接池限制
    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT_REQUESTS,
        max_keepalive_connections=MAX_CONCURRENT_REQUESTS
    )

    async with httpx.AsyncClient(limits=limits, timeout=REQUEST_TIMEOUT) as client:
        tasks = [
            classify_statement(stmt, system_prompt, api_client, client)
            for stmt in statements
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 保存结果
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"任务异常: {r}")
                continue

            save_result(r, output_file)


# ==================== 评估函数 ====================
def evaluate_results(result_dir: Path) -> Dict:
    """评估结果"""

    # 加载所有数据
    all_data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_data.append(json.loads(line.strip()))
            except:
                continue

    # 构建ground truth
    ground_truth = {}
    for item in all_data:
        statement = item.get("statement")
        label = item.get("label", [])
        if statement and len(label) >= 2:
            ground_truth[statement] = (label[0], label[1])

    # 加载预测结果
    result_file = result_dir / "llm_unified_full_results.jsonl"
    predictions = {}

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")
                if not statement or statement in predictions:
                    continue

                json_result = item.get("json")
                if not json_result:
                    result_str = item.get("result", "")
                    json_result = parse_json_from_response(result_str)

                if json_result and isinstance(json_result, dict):
                    predictions[statement] = json_result
            except:
                continue

    # 计算指标
    import numpy as np

    def compute_metrics(y_true, y_pred):
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
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }

    # 必要性
    nec_y_true, nec_y_pred = [], []
    for stmt, (nec_true, _) in ground_truth.items():
        if stmt in predictions:
            nec_y_true.append(nec_true)
            nec_y_pred.append(int(predictions[stmt].get("has_necessity_violation", False)))

    # 模糊性
    amb_y_true, amb_y_pred = [], []
    for stmt, (_, amb_true) in ground_truth.items():
        if stmt in predictions:
            amb_y_true.append(amb_true)
            amb_y_pred.append(int(predictions[stmt].get("has_ambiguity_violation", False)))

    nec_metrics = compute_metrics(nec_y_true, nec_y_pred) if nec_y_true else {}
    amb_metrics = compute_metrics(amb_y_true, amb_y_pred) if amb_y_true else {}

    return {
        "model_id": MODEL_ID,
        "total_samples": len(ground_truth),
        "matched_samples": len(predictions),
        "necessity": nec_metrics,
        "ambiguity": amb_metrics
    }


# ==================== 主程序 ====================
async def main():
    print("="*70)
    print("LLM统一分类器 - 完整数据集")
    print("="*70)
    print(f"模型: {MODEL_ID}")
    print(f"最大并发: {MAX_CONCURRENT_REQUESTS}")
    print(f"数据源: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*70)

    # 加载提示词
    print("\n加载提示词...")
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        if '=' in content:
            content = content.split('=', 1)[1].strip()
            if content.startswith('"""') or content.startswith("'''"):
                content = content[3:]
            if content.endswith('"""') or content.endswith("'''"):
                content = content[:-3]
        system_prompt = content.strip()
    print("提示词已加载")

    # 加载数据
    print("\n加载数据...")
    all_data = load_all_data()
    print(f"数据总量: {len(all_data)} 条")

    # 检查已处理数据
    output_file = OUTPUT_DIR / "llm_unified_full_results.jsonl"
    processed = get_processed_statements(output_file)
    to_process = [d for d in all_data if d.get("statement") not in processed]

    print(f"已处理: {len(processed)} 条")
    print(f"待处理: {len(to_process)} 条")

    if len(to_process) == 0:
        print("\n所有数据已处理，进行评估...")
        report = evaluate_results(OUTPUT_DIR)

        print("\n" + "="*70)
        print("评估结果")
        print("="*70)

        if report.get("necessity"):
            m = report["necessity"]
            print(f"\n必要性维度:")
            print(f"  准确率: {m['accuracy']:.4f}")
            print(f"  精确率: {m['precision']:.4f}")
            print(f"  召回率: {m['recall']:.4f}")
            print(f"  F1分数: {m['f1']:.4f}")

        if report.get("ambiguity"):
            m = report["ambiguity"]
            print(f"\n模糊性维度:")
            print(f"  准确率: {m['accuracy']:.4f}")
            print(f"  精确率: {m['precision']:.4f}")
            print(f"  召回率: {m['recall']:.4f}")
            print(f"  F1分数: {m['f1']:.4f}")

        report_file = OUTPUT_DIR / "evaluation_report_full.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n评估报告已保存: {report_file}")

        return

    # 创建API客户端
    api_client = DashScopeClient(
        api_key=DASHSCOPE_API_KEY,
        model_id=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    # 处理数据
    print(f"\n开始处理...")
    start_time = time.time()

    # 批处理
    batch_size = MAX_CONCURRENT_REQUESTS
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        statements = [d.get("statement") for d in batch if d.get("statement")]

        await process_batch(statements, system_prompt, api_client, output_file)

        processed_count = len(processed) + i + len(statements)
        print(f"\r进度: {processed_count}/{len(all_data)} ({processed_count/len(all_data)*100:.1f}%)", end='', flush=True)

    elapsed = time.time() - start_time

    print(f"\n\n完成! 总数: {len(to_process)} | 耗时: {elapsed:.1f}秒 | 平均: {elapsed/len(to_process):.2f}秒/条")

    # 评估
    print("\n" + "="*70)
    print("开始评估...")
    print("="*70)

    report = evaluate_results(OUTPUT_DIR)

    print("\n" + "="*70)
    print("评估结果")
    print("="*70)

    if report.get("necessity"):
        m = report["necessity"]
        print(f"\n必要性维度:")
        print(f"  准确率: {m['accuracy']:.4f}")
        print(f"  精确率: {m['precision']:.4f}")
        print(f"  召回率: {m['recall']:.4f}")
        print(f"  F1分数: {m['f1']:.4f}")

    if report.get("ambiguity"):
        m = report["ambiguity"]
        print(f"\n模糊性维度:")
        print(f"  准确率: {m['accuracy']:.4f}")
        print(f"  精确率: {m['precision']:.4f}")
        print(f"  召回率: {m['recall']:.4f}")
        print(f"  F1分数: {m['f1']:.4f}")

    # 保存评估报告
    report_file = OUTPUT_DIR / "evaluation_report_full.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评估报告已保存: {report_file}")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断，结果已保存")
    except Exception as e:
        logger.error(f"失败: {e}")
        import traceback
        traceback.print_exc()
