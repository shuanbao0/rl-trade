"""
数据模块核心组件
包含数据管理器和异常定义
"""

from .data_manager import DataManager
from .exceptions import *

__all__ = [
    'DataManager',
    # 异常类
    'DataSourceError',
    'DataInsufficientError',
    'AllDataSourcesFailedError',
    'DataProcessingError'
]