# -*- coding: utf-8 -*-
"""
批量运行两个独立任务的训练

训练必要性分类模型和表述模糊分类模型
"""

import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger(__name__)


def run_training(script_name, task_name):
    """运行单个训练脚本"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练: {task_name}")
    logger.info(f"{'='*60}\n")

    script_path = project_root / "experiments" / "baseline" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root)
    )

    if result.returncode == 0:
        logger.info(f"✓ {task_name} 训练完成")
    else:
        logger.error(f"✗ {task_name} 训练失败")
        return False

    return True


def main():
    """批量运行所有训练"""
    logger.info(f"\n{'='*60}")
    logger.info(f"批量训练: 必要性分类 + 表述模糊分类")
    logger.info(f"{'='*60}\n")

    tasks = [
        ("train_necessity.py", "必要性分类模型"),
        ("train_ambiguity.py", "表述模糊分类模型"),
    ]

    results = {}
    for script, name in tasks:
        results[name] = run_training(script, name)

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"训练任务完成总结")
    logger.info(f"{'='*60}")

    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"{status}: {name}")

    logger.info(f"\n模型保存位置:")
    logger.info(f"  必要性模型: results/models/roberta_necessity_classifier/")
    logger.info(f"  表述模糊模型: results/models/roberta_ambiguity_classifier/")


if __name__ == "__main__":
    main()
