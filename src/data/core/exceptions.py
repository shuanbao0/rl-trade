"""
数据管理模块异常类定义
"""

from typing import List, Optional


class DataSourceError(Exception):
    """数据源错误基类"""
    
    def __init__(self, message: str, source: str = None, symbol: str = None, details: dict = None):
        """
        初始化数据源错误
        
        Args:
            message: 错误消息
            source: 数据源名称
            symbol: 交易符号
            details: 额外错误详情
        """
        super().__init__(message)
        self.source = source
        self.symbol = symbol
        self.details = details or {}
        
    def __str__(self):
        """返回错误描述"""
        parts = [super().__str__()]
        
        if self.source:
            parts.append(f"Source: {self.source}")
        
        if self.symbol:
            parts.append(f"Symbol: {self.symbol}")
            
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
            
        return " | ".join(parts)


class DataSourceConnectionError(DataSourceError):
    """数据源连接错误"""
    pass


class DataSourceAuthenticationError(DataSourceError):
    """数据源认证错误"""
    pass


class DataSourceConfigurationError(DataSourceError):
    """数据源配置错误"""
    pass


class DataSourceRateLimitError(DataSourceError):
    """数据源速率限制错误"""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        """
        初始化速率限制错误
        
        Args:
            message: 错误消息
            retry_after: 建议重试等待时间（秒）
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class DataSourceNotSupportedError(DataSourceError):
    """数据源不支持错误"""
    pass


class DataValidationError(DataSourceError):
    """数据验证错误"""
    
    def __init__(self, message: str, validation_issues: List[str] = None, **kwargs):
        """
        初始化数据验证错误
        
        Args:
            message: 错误消息
            validation_issues: 验证问题列表
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)
        self.validation_issues = validation_issues or []


class DataInsufficientError(DataSourceError):
    """数据不足错误"""
    
    def __init__(self, message: str, expected_records: int = None, actual_records: int = None, **kwargs):
        """
        初始化数据不足错误
        
        Args:
            message: 错误消息
            expected_records: 期望记录数
            actual_records: 实际记录数
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)
        self.expected_records = expected_records
        self.actual_records = actual_records


class DataSourceCompatibilityError(DataSourceError):
    """数据源兼容性错误"""
    
    def __init__(self, message: str, compatibility_score: float = None, requirements: dict = None, **kwargs):
        """
        初始化兼容性错误
        
        Args:
            message: 错误消息
            compatibility_score: 兼容性评分
            requirements: 兼容性要求
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)
        self.compatibility_score = compatibility_score
        self.requirements = requirements or {}


class AllDataSourcesFailedError(DataSourceError):
    """所有数据源都失败的错误"""
    
    def __init__(self, message: str, failed_sources: List[str] = None, source_errors: dict = None, **kwargs):
        """
        初始化所有数据源失败错误
        
        Args:
            message: 错误消息
            failed_sources: 失败的数据源列表
            source_errors: 各数据源的具体错误
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)
        self.failed_sources = failed_sources or []
        self.source_errors = source_errors or {}


class CacheError(Exception):
    """缓存错误基类"""
    pass


class CacheCorruptedError(CacheError):
    """缓存损坏错误"""
    pass


class CacheExpiredError(CacheError):
    """缓存过期错误"""
    pass


class CacheStorageError(CacheError):
    """缓存存储错误"""
    pass


class MarketTypeDetectionError(Exception):
    """市场类型检测错误"""
    pass


class RoutingConfigurationError(Exception):
    """路由配置错误"""
    pass