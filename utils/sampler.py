# -*- coding: utf-8 -*-
"""
采样工具��块

提供从分类结果中按指定比例采样的功能
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SchemaSampler:
    """
    基于Schema的采样器

    根据指定的schema字段和目标比例，从数据中采样
    """

    def __init__(
        self,
        schema: List[str],  # 用于分组的字段列表
        input_file: Path,   # 输入文件路径
        output_file: Path,  # 输出文件路径
        sample_size: int,   # 总采样数
        ratios: Optional[List[float]] = None,  # 各组比例，默认为均匀分布
        random_seed: int = 42
    ):
        """
        初始化采样器

        Args:
            schema: 用于分组的字段名列表，如 ["has_necessity_violation", "has_ambiguity_violation"]
            input_file: 输入文件路径（JSONL格式）
            output_file: 输出文件路径
            sample_size: 总采样数
            ratios: 各组的比例列表，默认为均匀分布（None）
            random_seed: 随机种子
        """
        self.schema = schema
        self.input_file = input_file
        self.output_file = output_file
        self.sample_size = sample_size
        self.ratios = ratios
        self.random_seed = random_seed

        # 设置随机种子
        random.seed(random_seed)

        # 数据存储
        self.data = []
        self.groups = defaultdict(list)
        self.group_keys = []

    def load_data(self, key_func: Optional[Callable] = None):
        """
        加载数据并按schema分组

        Args:
            key_func: 自定义分组函数，接收item返回group_key。
                     如果为None，则使用schema字段值组合作为key
        """
        logger.info(f"加载数据: {self.input_file}")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)

                        # 生成分组key
                        if key_func:
                            group_key = key_func(item)
                        else:
                            # 默认使用schema字段的值组合
                            group_key = tuple(str(item.get(field)) for field in self.schema)

                        self.groups[group_key].append(item)

                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON失败: {e}")

        logger.info(f"加载数据完成，共 {len(self.data)} 条")

        # 记录所有分组key
        self.group_keys = list(self.groups.keys())
        logger.info(f"分组数: {len(self.group_keys)}")

        # 打印分组统计
        for key, items in sorted(self.groups.items()):
            logger.info(f"  {key}: {len(items)} 条")

        return self

    def calculate_sample_counts(self):
        """
        计算每组应该采样的数量

        Returns:
            dict: {group_key: sample_count}
        """
        num_groups = len(self.group_keys)

        # 计算每组采样数
        if self.ratios is None:
            # 默认均匀分布
            base_count = self.sample_size // num_groups
            remainder = self.sample_size % num_groups

            sample_counts = {}
            for i, key in enumerate(self.group_keys):
                count = base_count + (1 if i < remainder else 0)
                sample_counts[key] = count
        else:
            # 使用指定比例
            if len(self.ratios) != num_groups:
                raise ValueError(f"比例数量({len(self.ratios)})与分组数({num_groups})不匹配")

            sample_counts = {}
            for key, ratio in zip(self.group_keys, self.ratios):
                count = int(self.sample_size * ratio)
                sample_counts[key] = count

        # 检查每组是否有足够样本
        for key, count in sample_counts.items():
            available = len(self.groups[key])
            if available < count:
                logger.warning(f"分组 {key} 样本不足: 需要 {count}, 可用 {available}")
                sample_counts[key] = available

        total_available = sum(sample_counts.values())
        if total_available < self.sample_size:
            logger.warning(f"总采样数不足: 目标 {self.sample_size}, 实际可采样 {total_available}")

        return sample_counts

    def sample(self) -> List[Dict]:
        """
        执行采样

        Returns:
            采样后的数据列表
        """
        if not self.groups:
            raise ValueError("请先调用 load_data() 加载数据")

        # 计算每组采样数
        sample_counts = self.calculate_sample_counts()

        logger.info(f"\n采样计划:")
        for key, count in sample_counts.items():
            logger.info(f"  {key}: {count} 条")

        # 执行采样
        sampled_data = []
        for key, count in sample_counts.items():
            if count > 0:
                group_data = self.groups[key]
                sampled = random.sample(group_data, min(count, len(group_data)))
                sampled_data.extend(sampled)
                logger.info(f"从 {key} 采样 {len(sampled)} 条")

        logger.info(f"\n总采样数: {len(sampled_data)}")

        return sampled_data

    def save_results(self, sampled_data: Optional[List[Dict]] = None):
        """
        保存采样结果

        Args:
            sampled_data: 采样数据，如果为None则重新采样
        """
        if sampled_data is None:
            sampled_data = self.sample()

        # 确保输出目录存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 保存为JSONL格式
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"采样结果已保存到: {self.output_file}")

        return sampled_data

    def get_statistics(self, sampled_data: List[Dict]) -> Dict[str, Any]:
        """
        获取采样统计信息

        Args:
            sampled_data: 采样后的数据

        Returns:
            统计信息字典
        """
        # 统计各组的实际采样数
        sampled_groups = defaultdict(int)
        for item in sampled_data:
            key = tuple(str(item.get(field)) for field in self.schema)
            sampled_groups[key] += 1

        stats = {
            'total_sampled': len(sampled_data),
            'target_total': self.sample_size,
            'schema': self.schema,
            'group_counts': dict(sorted(sampled_groups.items())),
            'sampling_ratios': {k: v / len(sampled_data) for k, v in sampled_groups.items()}
        }

        return stats

    def run(self) -> tuple[List[Dict], Dict[str, Any]]:
        """
        运行完整的采样流程

        Returns:
            (采样数据, 统计信息)
        """
        # 加载数据
        self.load_data()

        # 采样
        sampled_data = self.sample()

        # 保存结果
        self.save_results(sampled_data)

        # 获取统计信息
        stats = self.get_statistics(sampled_data)

        logger.info(f"\n采样统计:")
        logger.info(f"  目标总数: {stats['target_total']}")
        logger.info(f"  实际采样: {stats['total_sampled']}")
        logger.info(f"  采样比例:")
        for key, ratio in stats['sampling_ratios'].items():
            logger.info(f"    {key}: {ratio:.2%}")

        return sampled_data, stats


def stratified_sample(
    schema: List[str],
    input_file: str,
    output_file: str,
    sample_size: int,
    ratios: Optional[List[float]] = None,
    random_seed: int = 42
) -> tuple[List[Dict], Dict[str, Any]]:
    """
    便捷函数：执行分层采样

    Args:
        schema: 用于分组的字段列表，如 ["necessity", "ambiguity"]
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_size: 总采样数
        ratios: 各组比例，默认为均匀分布
        random_seed: 随机种子

    Returns:
        (采样数据列表, 统计信息字典)

    Example:
        # 从LLM分类结果中按1:1:1:1采样100条
        data, stats = stratified_sample(
            schema=["necessity_pred", "ambiguity_pred"],
            input_file="results/predictions/llm_results.jsonl",
            output_file="results/predictions/llm_results_sampled.jsonl",
            sample_size=100
        )
    """
    sampler = SchemaSampler(
        schema=schema,
        input_file=Path(input_file),
        output_file=Path(output_file),
        sample_size=sample_size,
        ratios=ratios,
        random_seed=random_seed
    )

    return sampler.run()


def sample_from_llm_results(
    input_file: str,
    output_file: str,
    sample_size: int = 200,
    random_seed: int = 42
) -> tuple[List[Dict], Dict[str, Any]]:
    """
    从LLM分类结果中采样（1:1:1:1比例）

    按照两个维度的分类结果组合进行等比例采样：
    - (正常, 正常)
    - (正常, 违规)
    - (违规, 正常)
    - (违规, 违规)

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_size: 总采样数
        random_seed: 随机种子

    Returns:
        (采样数据列表, 统计信息字典)

    Example:
        # 采样200条，每种组合50条
        data, stats = sample_from_llm_results(
            input_file="results/predictions/llm_combined_results.jsonl",
            output_file="results/predictions/llm_sampled_200.jsonl",
            sample_size=200
        )
    """
    # 定义分组函数：从结果中提取预测标签
    def extract_labels(item):
        # 根据结果格式提取
        if 'necessity' in item and 'ambiguity' in item:
            # 独立分类器合并结果格式
            nec = item['necessity'].get('has_violation', False)
            amb = item['ambiguity'].get('has_violation', False)
        elif 'has_necessity_violation' in item and 'has_ambiguity_violation' in item:
            # 统一分类器格式
            nec = item.get('has_necessity_violation', False)
            amb = item.get('has_ambiguity_violation', False)
        else:
            # 尝试解析result字段
            try:
                result = json.loads(item.get('result', '{}'))
                nec = result.get('has_necessity_violation', False)
                amb = result.get('has_ambiguity_violation', False)
            except:
                nec = False
                amb = False

        # 转换为字符串标签
        return ('违规' if nec else '正常', '违规' if amb else '清晰')

    sampler = SchemaSampler(
        schema=['necessity', 'ambiguity'],  # 仅用于显示
        input_file=Path(input_file),
        output_file=Path(output_file),
        sample_size=sample_size,
        ratios=None,  # 默认均匀分布（1:1:1:1）
        random_seed=random_seed
    )

    # 使用自定义分组函数加载数据
    sampler.load_data(key_func=extract_labels)

    # 采样并保存
    sampled_data = sampler.sample()
    sampler.save_results(sampled_data)
    stats = sampler.get_statistics(sampled_data)

    logger.info(f"\nLLM分类结果采样统计:")
    logger.info(f"  目标总数: {sample_size}")
    logger.info(f"  实际采样: {len(sampled_data)}")
    logger.info(f"  各组采样数:")
    for key, count in stats['group_counts'].items():
        logger.info(f"    {key}: {count} 条")

    return sampled_data, stats


if __name__ == "__main__":
    import sys

    # 示例：从LLM分类结果中采样
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "sampled_results.jsonl"
        sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 200

        logging.basicConfig(level=logging.INFO)

        data, stats = sample_from_llm_results(
            input_file=input_file,
            output_file=output_file,
            sample_size=sample_size
        )

        print(f"\n✓ 采样完成！")
        print(f"  输入文件: {input_file}")
        print(f"  输出文件: {output_file}")
        print(f"  采样数量: {len(data)}")
    else:
        print("用法: python sampler.py <input_file> [output_file] [sample_size]")
        print("示例: python sampler.py results.jsonl sampled.jsonl 200")
