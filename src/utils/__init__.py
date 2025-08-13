"""
通用工具模块
负责日志配置、配置管理、常用函数等功能
"""

from .logger import setup_logger
from .config import Config

__all__ = ['setup_logger', 'Config'] 