# -*- coding: utf-8 -*-
"""
运行所有分类器并合并结果

同时运行三个分类器：
1. 统一分类器 (classification_prompt.md)
2. 必要性独立分类器 (necessity_violation_prompt.md)
3. 表述模糊独立分类器 (ambiguity_violation_prompt.md)

然后合并独立分类器的结果，生成统一格式的输出。
"""

import os
import sys
import json
import asyncio
import platform
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)


async def run_classifier(script_path: Path, name: str) -> Dict:
    """运行单个分类器"""
    logger.info(f"开始运行: {name}")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    if result.returncode == 0:
        logger.info(f"{name} 运行成功")
        return {"name": name, "success": True}
    else:
        logger.error(f"{name} 运行失败: {result.stderr}")
        return {"name": name, "success": False, "error": result.stderr}


def read_jsonl(file_path: Path) -> List[Dict]:
    """读取JSONL文件"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON失败: {e}")
    return results


def parse_llm_result(content: str) -> Dict:
    """解析LLM返回的JSON结果"""
    try:
        # 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取JSON片段
        import re
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # 解析失败，返回空字典
        return {}


def merge_independent_results(
    necessity_file: Path,
    ambiguity_file: Path,
    output_file: Path
):
    """合并独立分类器的结果"""
    logger.info(f"合并独立分类器结果...")

    # 读取两个文件
    necessity_results = read_jsonl(necessity_file)
    ambiguity_results = read_jsonl(ambiguity_file)

    logger.info(f"必要性结果: {len(necessity_results)} 条")
    logger.info(f"表述模糊结果: {len(ambiguity_results)} 条")

    # 按statement建立索引
    necessity_dict = {}
    for item in necessity_results:
        stmt = item.get("statement")
        if stmt:
            necessity_dict[stmt] = item

    ambiguity_dict = {}
    for item in ambiguity_results:
        stmt = item.get("statement")
        if stmt:
            ambiguity_dict[stmt] = item

    # 合并结果
    merged = []
    all_statements = set(necessity_dict.keys()) | set(ambiguity_dict.keys())

    for stmt in all_statements:
        nec_result = necessity_dict.get(stmt, {})
        amb_result = ambiguity_dict.get(stmt, {})

        # 解析结果
        nec_json = parse_llm_result(nec_result.get("result", "{}"))
        amb_json = parse_llm_result(amb_result.get("result", "{}"))

        merged_item = {
            "statement": stmt,
            "necessity": {
                "has_violation": nec_json.get("has_necessity_violation", False),
                "type": nec_json.get("necessity_type"),
                "reason": nec_json.get("reason", ""),
                "raw_response": nec_result.get("result", "")
            },
            "ambiguity": {
                "has_violation": amb_json.get("has_ambiguity_violation", False),
                "type": amb_json.get("ambiguity_type"),
                "reason": amb_json.get("reason", ""),
                "raw_response": amb_result.get("result", "")
            }
        }
        merged.append(merged_item)

    # 写入合并结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"合并完成! 共 {len(merged)} 条结果")
    logger.info(f"合并结果已保存到: {output_file}")

    return {
        "total": len(merged),
        "necessity_only": len([s for s in all_statements if s in necessity_dict and s not in ambiguity_dict]),
        "ambiguity_only": len([s for s in all_statements if s in ambiguity_dict and s not in necessity_dict]),
        "both": len([s for s in all_statements if s in necessity_dict and s in ambiguity_dict])
    }


def compare_results(
    unified_file: Path,
    merged_file: Path,
    output_file: Optional[Path] = None
):
    """对比统一分类器和独立分类器的结果"""
    logger.info(f"对比统一分类器和独立分类器的结果...")

    unified_results = read_jsonl(unified_file)
    merged_results = read_jsonl(merged_file)

    logger.info(f"统一分类器结果: {len(unified_results)} 条")
    logger.info(f"独立分类器合并结果: {len(merged_results)} 条")

    # 解析统一分类器结果
    unified_dict = {}
    for item in unified_results:
        stmt = item.get("statement")
        if stmt:
            unified_json = parse_llm_result(item.get("result", "{}"))
            unified_dict[stmt] = unified_json

    # 对比分析
    comparisons = []
    agreement_necessity = 0
    agreement_ambiguity = 0
    agreement_both = 0
    total = 0

    for item in merged_results:
        stmt = item.get("statement")
        if stmt in unified_dict:
            total += 1
            unified = unified_dict[stmt]
            necessity = item.get("necessity", {})
            ambiguity = item.get("ambiguity", {})

            nec_agree = str(unified.get("has_necessity_violation")) == str(necessity.get("has_violation"))
            amb_agree = str(unified.get("has_ambiguity_violation")) == str(ambiguity.get("has_violation"))

            if nec_agree:
                agreement_necessity += 1
            if amb_agree:
                agreement_ambiguity += 1
            if nec_agree and amb_agree:
                agreement_both += 1

            comparisons.append({
                "statement": stmt,
                "unified_necessity": unified.get("has_necessity_violation"),
                "independent_necessity": necessity.get("has_violation"),
                "necessity_agree": nec_agree,
                "unified_ambiguity": unified.get("has_ambiguity_violation"),
                "independent_ambiguity": ambiguity.get("has_violation"),
                "ambiguity_agree": amb_agree
            })

    # 打印统计
    logger.info(f"\n{'='*60}")
    logger.info(f"结果对比统计 (共 {total} 条)")
    logger.info(f"{'='*60}")
    logger.info(f"必要性一致: {agreement_necessity}/{total} ({agreement_necessity/total*100:.2f}%)")
    logger.info(f"表述模糊一致: {agreement_ambiguity}/{total} ({agreement_ambiguity/total*100:.2f}%)")
    logger.info(f"两者都一致: {agreement_both}/{total} ({agreement_both/total*100:.2f}%)")
    logger.info(f"{'='*60}\n")

    # 保存对比结果
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        comparison_report = {
            "total_compared": total,
            "necessity_agreement": {
                "count": agreement_necessity,
                "rate": agreement_necessity / total if total > 0 else 0
            },
            "ambiguity_agreement": {
                "count": agreement_ambiguity,
                "rate": agreement_ambiguity / total if total > 0 else 0
            },
            "both_agreement": {
                "count": agreement_both,
                "rate": agreement_both / total if total > 0 else 0
            },
            "detailed_comparisons": comparisons
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, ensure_ascii=False, indent=2)
        logger.info(f"对比结果已保存到: {output_file}")

    return {
        "total": total,
        "necessity_agreement_rate": agreement_necessity / total if total > 0 else 0,
        "ambiguity_agreement_rate": agreement_ambiguity / total if total > 0 else 0,
        "both_agreement_rate": agreement_both / total if total > 0 else 0
    }


async def run_all():
    """运行所有分类器"""
    logger.info(f"{'='*60}")
    logger.info(f"开始运行所有分类器")
    logger.info(f"{'='*60}")

    # 输出目录
    output_dir = project_root / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行三个分类器
    scripts = [
        (project_root / "experiments" / "llm_prompting" / "classify_unified.py", "统一分类器"),
        (project_root / "experiments" / "llm_prompting" / "classify_necessity.py", "必要性独立分类器"),
        (project_root / "experiments" / "llm_prompting" / "classify_ambiguity.py", "表述模糊独立分类器"),
    ]

    tasks = [run_classifier(script, name) for script, name in scripts]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result["success"]:
            logger.info(f"✓ {result['name']} 完成")
        else:
            logger.error(f"✗ {result['name']} 失败")

    # 合并独立分类器结果
    logger.info(f"\n{'='*60}")
    logger.info(f"合并独立分类器结果")
    logger.info(f"{'='*60}")

    merge_stats = merge_independent_results(
        necessity_file=output_dir / "llm_necessity_results.jsonl",
        ambiguity_file=output_dir / "llm_ambiguity_results.jsonl",
        output_file=output_dir / "llm_independent_merged.jsonl"
    )

    # 对比结果
    logger.info(f"\n{'='*60}")
    logger.info(f"对比统一分类器和独立分类器结果")
    logger.info(f"{'='*60}")

    comparison_stats = compare_results(
        unified_file=output_dir / "llm_unified_results.jsonl",
        merged_file=output_dir / "llm_independent_merged.jsonl",
        output_file=output_dir / "classifier_comparison.json"
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"所有任务完成!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_all())
