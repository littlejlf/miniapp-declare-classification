# -*- coding: utf-8 -*-
"""
日志配置工具模块

提供统一的日志配置和管理功能。
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器"""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        console_output: bool = True
    ) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志记录器名称
            log_level: 日志级别
            log_file: 日志文件路径，如果为None则不写入文件
            console_output: 是否输出到控制台

        Returns:
            配置好的日志记录器
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # 清除已有的处理器
        logger.handlers.clear()

        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件输出
        if log_file:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 防止日志传播到根日志记录器
        logger.propagate = False

        cls._loggers[name] = logger
        return logger

    @classmethod
    def setup_experiment_logger(
        cls,
        experiment_name: str,
        log_dir: str = "results/logs",
        console_output: bool = True
    ) -> logging.Logger:
        """
        为实验设置日志记录器

        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
            console_output: 是否输出到控制台

        Returns:
            配置好的日志记录器
        """
        # 确保日志目录存在
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

        return cls.get_logger(
            name=experiment_name,
            log_level=logging.INFO,
            log_file=log_file,
            console_output=console_output
        )


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器的便捷函数

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return Logger.get_logger(name)


# 默认日志配置
def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
):
    """
    配置根日志记录器

    Args:
        level: 日志级别
        format_string: 自定义格式字符串
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# 为第三方库设置日志级别
def set_library_log_levels():
    """设置常用第三方库的日志级别，避免过多输出"""
    logging.getLogger('backoff').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('dashscope').setLevel(logging.WARNING)
