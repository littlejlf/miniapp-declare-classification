# -*- coding: utf-8 -*-
"""
LLM分类实验一键运行脚本

功能：
1. 运行三种分类器（统一、必要性独立、表述模糊独立）
2. 合并独立分类器结果
3. 评估所有分类器性能
4. 对比分析
5. 采样指定数量数据

使用方法:
    # 设置API key
    export DASHSCOPE_API_KEY='your-api-key'

    # 运行完整实验
    python run_llm_experiment.py

    # 或者只运行部分步骤
    python run_llm_experiment.py --steps unified evaluate sample
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import get_logger, set_library_log_levels

# 配置日志
set_library_log_levels()
logger = get_logger(__name__)


# ==================== 配置 ====================
class Config:
    """实验配置"""

    # 路径配置
    PROJECT_ROOT = project_root
    DATA_DIR = project_root / "data" / "raw"
    OUTPUT_DIR = project_root / "results" / "predictions"
    EVALUATION_DIR = project_root / "results" / "predictions" / "evaluation"

    # 输入文件
    INPUT_FILE = DATA_DIR / "aggregate_datas_label.jsonl"

    # 输出文件
    UNIFIED_OUTPUT = OUTPUT_DIR / "llm_unified_results.jsonl"
    NECESSITY_OUTPUT = OUTPUT_DIR / "llm_necessity_results.jsonl"
    AMBIGUITY_OUTPUT = OUTPUT_DIR / "llm_ambiguity_results.jsonl"
    INDEPENDENT_MERGED = OUTPUT_DIR / "llm_independent_merged.jsonl"
    SAMPLED_OUTPUT = OUTPUT_DIR / "llm_sampled_200.jsonl"

    # 模型配置
    MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen-plus")
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "10"))
    CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "100"))

    # 采样配置
    SAMPLE_SIZE = 200
    SAMPLE_SEED = 42


# ==================== 工具函数 ====================
def print_step(step: int, total: int, title: str):
    """打印步骤标题"""
    print(f"\n{'='*70}")
    print(f"步骤 {step}/{total}: {title}")
    print(f"{'='*70}\n")


def run_command(cmd: List[str], description: str) -> bool:
    """运行命令并返回是否成功"""
    logger.info(f"执行: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        elapsed = time.time() - start_time
        logger.info(f"✓ {description} 完成 (耗时: {elapsed:.2f}秒)")

        if result.stdout:
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} 失败 (��时: {elapsed:.2f}秒)")
        logger.error(f"错误信息: {e.stderr}")
        return False


def check_api_key() -> bool:
    """检查API key是否设置"""
    if not Config.API_KEY:
        logger.error("请设置环境变量 DASHSCOPE_API_KEY")
        logger.error("示例: export DASHSCOPE_API_KEY='your-api-key'")
        return False

    logger.info(f"API Key已设置: {Config.API_KEY[:10]}...")
    return True


def check_input_file() -> bool:
    """检查输入文件是否存在"""
    if not Config.INPUT_FILE.exists():
        logger.error(f"输入文件不存在: {Config.INPUT_FILE}")
        return False

    # 统计文件行数
    with open(Config.INPUT_FILE, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)

    logger.info(f"输入文件: {Config.INPUT_FILE}")
    logger.info(f"样本数量: {line_count}")

    return True


def create_directories():
    """创建必要的目录"""
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {Config.OUTPUT_DIR}")


# ==================== 实验步骤 ====================
def step_run_unified_classifier():
    """步骤1: 运行统一分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "classify_unified.py")
    ]
    return run_command(cmd, "统一分类器")


def step_run_necessity_classifier():
    """步骤2: 运行必要性独立分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "classify_necessity.py")
    ]
    return run_command(cmd, "必要性独立分类器")


def step_run_ambiguity_classifier():
    """步骤3: 运行表述模糊独立分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "classify_ambiguity.py")
    ]
    return run_command(cmd, "表述模糊独立分类器")


def step_merge_independent_results():
    """步骤4: 合并独立分类器结果"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "run_all_classifications.py")
    ]
    return run_command(cmd, "合并独立分类器结果")


def step_evaluate_unified():
    """步骤5: 评估统一分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "evaluate_classification.py"),
        "--input", str(Config.UNIFIED_OUTPUT),
        "--labels", str(Config.INPUT_FILE),
        "--type", "unified",
        "--output", str(Config.EVALUATION_DIR)
    ]
    return run_command(cmd, "评估统一分类器")


def step_evaluate_independent():
    """步骤6: 评估独立分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "evaluate_classification.py"),
        "--input", str(Config.INDEPENDENT_MERGED),
        "--labels", str(Config.INPUT_FILE),
        "--type", "independent",
        "--output", str(Config.EVALUATION_DIR)
    ]
    return run_command(cmd, "评估独立分类器")


def step_compare_classifiers():
    """步骤7: 对比两种分类器"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "evaluate_classification.py"),
        "--input", str(Config.INDEPENDENT_MERGED),
        "--labels", str(Config.INPUT_FILE),
        "--type", "compare",
        "--output", str(Config.EVALUATION_DIR)
    ]
    return run_command(cmd, "对比分类器")


def step_sample_results(size: int = Config.SAMPLE_SIZE):
    """步骤8: 采样结果"""
    cmd = [
        sys.executable,
        str(project_root / "experiments" / "llm_prompting" / "sample_llm_results.py"),
        "--input", str(Config.UNIFIED_OUTPUT),
        "--output", str(Config.OUTPUT_DIR / f"llm_sampled_{size}.jsonl"),
        "--size", str(size),
        "--seed", str(Config.SAMPLE_SEED)
    ]
    return run_command(cmd, f"采样{size}条数据")


# ==================== 主流程 ====================
def run_full_experiment():
    """运行完整实验"""
    total_steps = 8

    print_step(1, total_steps, "运行统一分类器")
    if not step_run_unified_classifier():
        return False

    print_step(2, total_steps, "运行必要性独立分类器")
    if not step_run_necessity_classifier():
        return False

    print_step(3, total_steps, "运行表述模糊独立分类器")
    if not step_run_ambiguity_classifier():
        return False

    print_step(4, total_steps, "合并独立分类器结果")
    if not step_merge_independent_results():
        return False

    print_step(5, total_steps, "评估统一分类器")
    if not step_evaluate_unified():
        return False

    print_step(6, total_steps, "评估独立分类器")
    if not step_evaluate_independent():
        return False

    print_step(7, total_steps, "对比两种分类器")
    if not step_compare_classifiers():
        return False

    print_step(8, total_steps, f"采样{Config.SAMPLE_SIZE}条数据")
    if not step_sample_results(Config.SAMPLE_SIZE):
        return False

    return True


def run_experiment(steps: List[str], sample_size: int = None):
    """运行指定步骤"""
    step_map = {
        "unified": step_run_unified_classifier,
        "necessity": step_run_necessity_classifier,
        "ambiguity": step_run_ambiguity_classifier,
        "merge": step_merge_independent_results,
        "eval-unified": step_evaluate_unified,
        "eval-independent": step_evaluate_independent,
        "compare": step_compare_classifiers,
        "sample": lambda: step_sample_results(sample_size or Config.SAMPLE_SIZE),
    }

    total = len(steps)
    for idx, step in enumerate(steps, 1):
        if step not in step_map:
            logger.error(f"未知步骤: {step}")
            return False

        print_step(idx, total, f"执行: {step}")
        if not step_map[step]():
            return False

    return True


def print_summary():
    """打印实验总结"""
    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")
    print(f"\n结果文件位置: {Config.OUTPUT_DIR}")
    print(f"评估报告位置: {Config.EVALUATION_DIR}")

    # 列出生成的文件
    print(f"\n生成的文件:")
    for file in Config.OUTPUT_DIR.iterdir():
        size = file.stat().st_size if file.is_file() else 0
        print(f"  - {file.name} ({size:,} bytes)")

    # 列出评估报告
    if Config.EVALUATION_DIR.exists():
        print(f"\n评估报告:")
        for file in Config.EVALUATION_DIR.iterdir():
            size = file.stat().st_size if file.is_file() else 0
            print(f"  - {file.name} ({size:,} bytes)")

    print(f"\n{'='*70}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LLM分类实验一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 运行完整实验
  python run_llm_experiment.py

  # 运行指定步骤
  python run_llm_experiment.py --steps unified evaluate sample

  # 指定采样数量
  python run_llm_experiment.py --sample-size 100

可用步骤:
  unified       - 运行统一分类器
  necessity     - 运行必要性分类器
  ambiguity     - 运行表述模糊分类器
  merge         - 合并独立分类器结果
  eval-unified  - 评估统一分类器
  eval-independent - 评估独立分类器
  compare       - 对比两种分类器
  sample        - 采样数据
  evaluate      - 运行所有评估步骤
        """
    )

    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=None,
        help="指定要运行的步骤（默认运行全部）"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=Config.SAMPLE_SIZE,
        help=f"采样数量（默认: {Config.SAMPLE_SIZE}）"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API Key（也可通过环境变量DASHSCOPE_API_KEY设置）"
    )

    args = parser.parse_args()

    # 设置API key
    if args.api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key

    # 检查环境
    if not check_api_key():
        sys.exit(1)

    if not check_input_file():
        sys.exit(1)

    create_directories()

    # 打印配置
    logger.info(f"模型: {Config.MODEL_ID}")
    logger.info(f"并发数: {Config.CONCURRENCY}")
    logger.info(f"分块大小: {Config.CHUNK_SIZE}")

    # 运行实验
    start_time = time.time()

    try:
        if args.steps is None:
            # 运行完整实验
            success = run_full_experiment()
        else:
            # 运行指定步骤
            success = run_experiment(args.steps, args.sample_size)

        elapsed = time.time() - start_time

        if success:
            print_summary()
            logger.info(f"总耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
        else:
            logger.error("实验执行失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"实验执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
