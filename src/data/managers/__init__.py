"""
数据管理器模块
包含缓存、路由、批次下载等管理功能
"""

from .cache_manager import get_cache_manager, CacheKey
from .routing_manager import get_routing_manager
from .batch_downloader import BatchDownloader

__all__ = [
    'get_cache_manager',
    'CacheKey', 
    'get_routing_manager',
    'BatchDownloader'
]