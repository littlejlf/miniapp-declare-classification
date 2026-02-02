# -*- coding: utf-8 -*-
"""
配置加载工具模块

提供统一的配置加载和管理功能。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """配置管理类"""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        初始化配置管理器

        Args:
            config_dir: 配置文件目录，如果为None则使用默认目录
        """
        if config_dir is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)
        self.configs = {}

        # 加载所有配置文件
        self._load_all_configs()

    def _load_all_configs(self):
        """加载所有配置文件"""
        config_files = {
            "model": "model_config.yaml",
            "training": "training_config.yaml",
            "data": "data_config.yaml"
        }

        for key, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                self.configs[key] = self._load_yaml(config_path)
                logger.info(f"已加载配置: {filename}")
            else:
                logger.warning(f"配置文件不存在: {config_path}")
                self.configs[key] = {}

    def _load_yaml(self, yaml_path: Path) -> Dict[str, Any]:
        """
        加载YAML文件

        Args:
            yaml_path: YAML文件路径

        Returns:
            配置字典
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败 {yaml_path}: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键，如 'model.roberta.model_name'
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')

        # 先查找特定的配置文件
        config_key = keys[0]
        if config_key in self.configs:
            config = self.configs[config_key]
            keys = keys[1:]  # 移除第一个键
        else:
            # 如果没有指定配置文件，在所有配置中查找
            for config in self.configs.values():
                value = self._get_nested(config, keys)
                if value is not None:
                    return value
            return default

        return self._get_nested(config, keys, default)

    def _get_nested(self, config: Dict[str, Any], keys: list, default: Any = None) -> Any:
        """
        从嵌套字典中获取值

        Args:
            config: 配置字典
            keys: 键列表
            default: 默认值

        Returns:
            配置值
        """
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.configs.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.configs.get('training', {})

    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.configs.get('data', {})

    def update_from_env(self):
        """从环境变量更新配置"""
        # API Key
        if 'DASHSCOPE_API_KEY' in os.environ:
            if 'model' not in self.configs:
                self.configs['model'] = {}
            if 'llm' not in self.configs['model']:
                self.configs['model']['llm'] = {}
            self.configs['model']['llm']['api_key_env'] = os.environ['DASHSCOPE_API_KEY']

        # 模型ID
        if 'LLM_MODEL_ID' in os.environ:
            if 'model' not in self.configs:
                self.configs['model'] = {}
            if 'llm' not in self.configs['model']:
                self.configs['model']['llm'] = {}
            self.configs['model']['llm']['model_name'] = os.environ['LLM_MODEL_ID']

        logger.info("已从环境变量更新配置")


# 全局配置实例
_global_config = None


def get_config(reload: bool = False) -> Config:
    """
    获取全局配置实例

    Args:
        reload: 是否重新加载配置

    Returns:
        配置实例
    """
    global _global_config

    if _global_config is None or reload:
        _global_config = Config()
        _global_config.update_from_env()

    return _global_config
