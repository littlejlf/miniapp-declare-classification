# -*- coding: utf-8 -*-
"""
批量运行多个模型对250条筛选数据进行分类

结果文件命名: {samplesize}_{model_id}_{date}.jsonl
"""

import os
import sys
import json
import time
import asyncio
import platform
from pathlib import Path
from typing import List, Dict
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

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 配置 ====================
API_KEY = "sk-071feb0c2b074feabbac6677c5954ef8"
TEMPERATURE = 0.1
MAX_TOKENS = 2000

SAMPLE_SIZE = 250
SAMPLE_DATA_FILE = project_root / "results" / "predictions" / "qwen3-32b" / "filtered_250" / "filtered_250_balanced.jsonl"

MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_WAIT_MIN = 1.0
RETRY_WAIT_MAX = 10.0

PROMPT_FILE = project_root / "prompts" / "classification_prompt.md"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 加载模型列表 ====================
def load_model_list():
    """加载模型列表"""
    model_file = project_root / "experiments" / "llm_prompting" / "model_id.txt"
    with open(model_file, 'r', encoding='utf-8') as f:
        models = [line.strip() for line in f if line.strip()]
    return models


# ==================== 加载样本数据 ====================
def load_sample_data():
    """加载筛选后的250条数据"""
    samples = []
    with open(SAMPLE_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except:
                continue
    return samples


# ==================== 加载提示词 ====================
def load_prompt():
    """加载提示词"""
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        if '=' in content:
            content = content.split('=', 1)[1].strip()
            if content.startswith('"""') or content.startswith("'''"):
                content = content[3:]
            if content.endswith('"""') or content.endswith("'''"):
                content = content[:-3]
        return content.strip()


# ==================== DashScope API 调用 ====================
class DashScopeClient:
    """DashScope API 客户端"""

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self.api_key = api_key
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
    ) -> Dict:
        """调用 DashScope API"""

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
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
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


# ==================== 分类处理 ====================
def parse_json_from_response(content: str):
    """从响应中解析JSON"""
    try:
        return json.loads(content)
    except:
        if '```json' in content:
            content = content.split('```json', 1)[1]
        if '```' in content:
            content = content.split('```')[0].strip()
        try:
            return json.loads(content)
        except:
            return None


async def classify_samples(
    samples: List[Dict],
    model_id: str,
    system_prompt: str,
    output_file: Path
):
    """对样本进行分类"""

    api_client = DashScopeClient(model_id, API_KEY)

    limits = httpx.Limits(
        max_connections=MAX_CONCURRENT_REQUESTS,
        max_keepalive_connections=MAX_CONCURRENT_REQUESTS
    )

    async with httpx.AsyncClient(limits=limits, timeout=REQUEST_TIMEOUT) as client:
        total = len(samples)

        for idx, sample in enumerate(samples):
            statement = sample["statement"]

            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": statement}
            ]

            start_time = time.time()

            try:
                response = await api_client.call_api(client, messages)
                elapsed = time.time() - start_time

                # 解析响应
                if response.get("output") and response["output"].get("choices"):
                    content = response["output"]["choices"][0]["message"]["content"]

                    result = {
                        "statement": statement,
                        "label": sample["label"],
                        "model_id": model_id,
                        "classifier": "统一分类器",
                        "result": content,
                        "status": "pending",
                        "response_time": elapsed,
                        "timestamp": datetime.now().isoformat()
                    }

                    # 尝试解析JSON
                    parsed_json = parse_json_from_response(content)
                    if parsed_json:
                        result["json"] = parsed_json
                        result["status"] = "success"
                    else:
                        result["status"] = "format_error"
                        result["error"] = "JSON解析失败"

                else:
                    result = {
                        "statement": statement,
                        "label": sample["label"],
                        "model_id": model_id,
                        "classifier": "统一分类器",
                        "result": "",
                        "status": "failed",
                        "error": "响应格式错误",
                        "response_time": elapsed,
                        "timestamp": datetime.now().isoformat()
                    }

            except Exception as e:
                elapsed = time.time() - start_time
                result = {
                    "statement": statement,
                    "label": sample["label"],
                    "model_id": model_id,
                    "classifier": "统一分类器",
                    "result": "",
                    "status": "failed",
                    "error": str(e),
                    "response_time": elapsed,
                    "timestamp": datetime.now().isoformat()
                }

            # 保存结果
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            print(f"\r进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) | 响应: {elapsed:.2f}s", end='', flush=True)

        print(f"\n完成!")


# ==================== 评估函数 ====================
def evaluate_results(result_file: Path, model_id: str):
    """评估结果"""
    import numpy as np

    # 加载结果
    predictions = {}
    ground_truth = {}

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                statement = item.get("statement")

                # 保存真实标签
                if statement and statement not in ground_truth:
                    label = item.get("label", [])
                    if len(label) >= 2:
                        ground_truth[statement] = (label[0], label[1])

                # 保存预测
                if statement and statement not in predictions:
                    json_result = item.get("json")
                    if not json_result:
                        result_str = item.get("result", "")
                        json_result = parse_json_from_response(result_str)

                    if json_result and isinstance(json_result, dict):
                        predictions[statement] = json_result
            except:
                continue

    # 计算指标
    def compute_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = np.mean(y_true == y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
        }

    # 必要性
    nec_y_true, nec_y_pred = [], []
    amb_y_true, amb_y_pred = [], []

    for stmt in ground_truth:
        if stmt in predictions:
            nec_true, amb_true = ground_truth[stmt]
            pred = predictions[stmt]

            nec_y_true.append(nec_true)
            amb_y_true.append(amb_true)

            nec_y_pred.append(int(pred.get("has_necessity_violation", False)))
            amb_y_pred.append(int(pred.get("has_ambiguity_violation", False)))

    nec_metrics = compute_metrics(nec_y_true, nec_y_pred) if nec_y_true else {}
    amb_metrics = compute_metrics(amb_y_true, amb_y_pred) if amb_y_true else {}

    return {
        "model_id": model_id,
        "sample_size": len(ground_truth),
        "matched_samples": len(predictions),
        "necessity": nec_metrics,
        "ambiguity": amb_metrics
    }


# ==================== 主程序 ====================
async def main():
    print("="*70)
    print("批量运行多模型分类 - 250条筛选数据")
    print("="*70)

    # 加载模型列表
    models = load_model_list()
    print(f"\n模型列表 ({len(models)}个):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")

    # 加载样本数据
    samples = load_sample_data()
    print(f"\n样本数据: {len(samples)} 条")

    # 加载提示词
    system_prompt = load_prompt()
    print(f"提示词已加载")

    # 获取日期
    date_str = datetime.now().strftime("%Y%m%d")

    # 依次处理每个模型
    for model_id in models:
        print(f"\n{'='*70}")
        print(f"处理模型: {model_id}")
        print(f"{'='*70}")

        # 输出目录
        output_dir = project_root / "results" / "predictions" / model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 结果文件名: {samplesize}_{model_id}_{date}.jsonl
        output_file = output_dir / f"{SAMPLE_SIZE}_{model_id}_{date_str}.jsonl"

        # 检查是否已处理
        if output_file.exists():
            existing_lines = sum(1 for _ in open(output_file))
            if existing_lines >= len(samples):
                print(f"已存在结果: {output_file.name}")
                print(f"样本数: {existing_lines}")
                # 评估
                report = evaluate_results(output_file, model_id)
                print(f"必要性F1: {report['necessity'].get('f1', 0)*100:.2f}%")
                print(f"模糊性F1: {report['ambiguity'].get('f1', 0)*100:.2f}%")
                continue

        print(f"输出文件: {output_file.name}")

        # 运行分类
        start_time = time.time()
        await classify_samples(samples, model_id, system_prompt, output_file)
        elapsed = time.time() - start_time

        print(f"\n模型 {model_id} 完成!")
        print(f"耗时: {elapsed:.1f}秒 ({elapsed/len(samples):.2f}秒/条)")

        # 评估
        print(f"评估中...")
        report = evaluate_results(output_file, model_id)

        print(f"\n评估结果:")
        print(f"  必要性F1: {report['necessity'].get('f1', 0)*100:.2f}%")
        print(f"  模糊性F1: {report['ambiguity'].get('f1', 0)*100:.2f}%")

        # 保存评估报告
        report_file = output_dir / f"evaluation_report_{SAMPLE_SIZE}_{date_str}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"评估报告: {report_file.name}")

    print(f"\n{'='*70}")
    print(f"全部完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n被中断")
    except Exception as e:
        logger.error(f"失败: {e}")
        import traceback
        traceback.print_exc()
