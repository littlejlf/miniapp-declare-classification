# -*- coding: utf-8 -*-
"""
从LLM分类结果中按比例采样

按照两个维度（必要性、表述模糊）的分类结果组合进行等比例采样：
- (正常, 清晰)
- (正常, 模糊)
- (违规, 清晰)
- (违规, 模糊)
"""

import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.sampler import sample_from_llm_results
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="从LLM分类结果中按1:1:1:1比例采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 从统一分类结果中采样200条
  python sample_llm_results.py \\
      --input results/predictions/llm_unified_results.jsonl \\
      --output results/predictions/llm_sampled_200.jsonl \\
      --size 200

  # 从独立分类器合并结果中采样
  python sample_llm_results.py \\
      --input results/predictions/llm_independent_merged.jsonl \\
      --output results/predictions/llm_sampled_100.jsonl \\
      --size 100 \\
      --seed 123
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文件路径（JSONL格式）'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径（JSONL格式）'
    )

    parser.add_argument(
        '--size',
        type=int,
        default=200,
        help='总采样数（默认：200）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )

    args = parser.parse_args()

    logger.info(f"{'='*60}")
    logger.info(f"LLM分类结果采样")
    logger.info(f"{'='*60}")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"采样总数: {args.size}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"采样比例: 1:1:1:1（四种组合等比例）")
    logger.info(f"{'='*60}\n")

    # 执行采样
    try:
        data, stats = sample_from_llm_results(
            input_file=args.input,
            output_file=args.output,
            sample_size=args.size,
            random_seed=args.seed
        )

        # 打印详细统计
        print(f"\n{'='*60}")
        print(f"采样结果汇总")
        print(f"{'='*60}")
        print(f"目标总数: {stats['target_total']}")
        print(f"实际采样: {stats['total_sampled']}")
        print(f"\n各组采样数:")

        # 按顺序显示四个组合
        group_names = {
            ('正常', '清晰'): '正常-清晰',
            ('正常', '模糊'): '正常-模糊',
            ('违规', '清晰'): '违规-清晰',
            ('违规', '模糊'): '违规-模糊'
        }

        for key, name in group_names.items():
            count = stats['group_counts'].get(key, 0)
            ratio = stats['sampling_ratios'].get(key, 0)
            print(f"  {name}: {count} 条 ({ratio:.1%})")

        print(f"\n✓ 采样完成！")
        print(f"结果已保存到: {args.output}")

    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"采样失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
