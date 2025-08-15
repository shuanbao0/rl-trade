"""
数据管理模块
负责股票数据获取、验证、缓存等功能

重构后的模块结构：
- core/: 核心组件（DataManager, 异常定义）
- sources/: 数据源实现
- services/: 服务层（数据获取、验证、日期范围处理）
- managers/: 管理器（缓存、路由、批次下载）
- processors/: 处理器（市场检测、数据处理）
- advisors/: 智能建议器（时间范围建议、兼容性检查）
- config/: 配置文件
"""

# 核心组件
from .core import DataManager
from .core.exceptions import (
    DataSourceError,
    DataSourceConnectionError,
    DataSourceAuthenticationError,
    DataSourceConfigurationError,
    DataSourceRateLimitError,
    DataSourceNotSupportedError,
    DataValidationError,
    DataInsufficientError,
    DataSourceCompatibilityError,
    AllDataSourcesFailedError,
    CacheError,
    CacheCorruptedError,
    CacheExpiredError,
    CacheStorageError,
    MarketTypeDetectionError,
    RoutingConfigurationError
)

# 数据源
from .sources.base import DataSource, DataQuality, MarketType, DataInterval, MarketData, DataPeriod

# 管理器
from .managers import get_routing_manager, get_cache_manager, BatchDownloader

# 处理器
from .processors import MarketTypeDetector, MarketProcessorFactory

# 智能建议器
from .advisors import get_smart_time_advisor, get_time_compatibility_checker, get_compatibility_checker

# 服务层
from .services import DataFetcher, DataValidator, DateRangeFetcher

# 便利函数（向后兼容）
from .processors.market_detector import detect_market_type, detect_market_types

__all__ = [
    # 核心组件
    'DataManager',
    
    # 数据源相关
    'DataSource', 'DataQuality', 'MarketType', 'DataInterval', 'MarketData', 'DataPeriod',
    
    # 管理器
    'get_routing_manager', 'get_cache_manager', 'BatchDownloader',
    
    # 处理器 
    'MarketTypeDetector', 'detect_market_type', 'detect_market_types', 'MarketProcessorFactory',
    
    # 智能建议器
    'get_smart_time_advisor', 'get_time_compatibility_checker', 'get_compatibility_checker',
    
    # 服务层
    'DataFetcher', 'DataValidator', 'DateRangeFetcher',
    
    # 异常类
    'DataSourceError',
    'DataSourceConnectionError',
    'DataSourceAuthenticationError',
    'DataSourceConfigurationError',
    'DataSourceRateLimitError',
    'DataSourceNotSupportedError',
    'DataValidationError',
    'DataInsufficientError',
    'DataSourceCompatibilityError',
    'AllDataSourcesFailedError',
    'CacheError',
    'CacheCorruptedError',
    'CacheExpiredError',
    'CacheStorageError',
    'MarketTypeDetectionError',
    'RoutingConfigurationError'
] 