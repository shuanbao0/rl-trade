"""
数据服务模块

提供统一的数据下载、实时传输等服务接口
"""

# 原有服务
from .data_fetcher import DataFetcher
from .data_validator import DataValidator, DataValidationResult
from .date_range_fetcher import DateRangeFetcher

# 新增服务
from .download_models import (
    DownloadRequest,
    DownloadResult,
    MultiDownloadRequest,
    MultiDownloadResult,
    BatchDownloadRequest,
    BatchDownloadResult,
    RealtimeRequest,
    RealtimeStream
)

from .download_service import DownloadService
from .realtime_service import RealtimeService, get_realtime_service, DataPoint, StreamStatus

# 便捷访问函数
def get_download_service():
    """获取下载服务实例"""
    return DownloadService()

def download_single(symbol: str, **kwargs) -> DownloadResult:
    """
    快速单个下载
    
    Args:
        symbol: 股票代码
        **kwargs: 其他下载参数
        
    Returns:
        DownloadResult: 下载结果
    """
    request = DownloadRequest(symbol=symbol, **kwargs)
    service = get_download_service()
    return service.download_single(request)

def download_multiple(symbols: list, **kwargs) -> MultiDownloadResult:
    """
    快速多个下载
    
    Args:
        symbols: 股票代码列表
        **kwargs: 其他下载参数
        
    Returns:
        MultiDownloadResult: 多个下载结果
    """
    request = MultiDownloadRequest(symbols=symbols, **kwargs)
    service = get_download_service()
    return service.download_multiple(request)

__all__ = [
    # 原有服务
    'DataFetcher',
    'DataValidator', 
    'DataValidationResult',
    'DateRangeFetcher',
    
    # 数据模型
    'DownloadRequest',
    'DownloadResult', 
    'MultiDownloadRequest',
    'MultiDownloadResult',
    'BatchDownloadRequest',
    'BatchDownloadResult',
    'RealtimeRequest',
    'RealtimeStream',
    
    # 服务类
    'DownloadService',
    'RealtimeService',
    'DataPoint',
    'StreamStatus',
    
    # 便捷函数
    'get_download_service',
    'get_realtime_service',
    'download_single',
    'download_multiple',
]