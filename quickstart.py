#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本

用于在AutoDL上快速启动实验的便捷脚本。
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger(__name__)


def check_env():
    """检查环境配置"""
    errors = []

    # 检查API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        errors.append("未设置 DASHSCOPE_API_KEY 环境变量")

    # 检查数据文件
    data_dir = project_root / "data" / "raw"
    if not data_dir.exists():
        errors.append(f"数据目录不存在: {data_dir}")
    else:
        all_declares = data_dir / "all_declares.jsonl"
        labeled_data = data_dir / "aggregate_datas_label.jsonl"

        if not all_declares.exists():
            errors.append(f"数据文件不存在: {all_declares}")
        if not labeled_data.exists():
            errors.append(f"数据文件不存在: {labeled_data}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="小程序隐私分类实验快速启动")
    parser.add_argument(
        "experiment",
        choices=["baseline", "llm", "augment", "check"],
        help="实验类型或检查环境"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="LLM API并发数（默认10）"
    )

    args = parser.parse_args()

    # 检查环境
    if args.experiment == "check":
        logger.info("检查环境配置...")
        errors = check_env()
        if errors:
            logger.error("发现问题:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        else:
            logger.info("环境配置正常！")
            logger.info(f"项目根目录: {project_root}")
            logger.info(f"Python版本: {sys.version}")
            return

    # 检查必需的环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.error("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        logger.info("设置方法: export DASHSCOPE_API_KEY='your-key-here'")
        sys.exit(1)

    # 运行实验
    if args.experiment == "baseline":
        logger.info("启动 RoBERTa 基线模型训练...")
        os.system(f"python {project_root / 'experiments' / 'baseline' / 'train_roberta.py'}")

    elif args.experiment == "llm":
        logger.info("启动 LLM 提示词分类...")
        os.environ["LLM_CONCURRENCY"] = str(args.concurrency)
        os.system(f"python {project_root / 'experiments' / 'llm_prompting' / 'classify_async.py'}")

    elif args.experiment == "augment":
        logger.info("启动数据增强...")
        os.system(f"python {project_root / 'data_processing' / 'augment_data.py'}")


if __name__ == "__main__":
    main()
