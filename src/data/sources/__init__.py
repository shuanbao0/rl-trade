"""
多数据源架构模块

提供统一的金融数据源访问接口，支持：
- 股票数据 (YFinance)  
- 外汇分钟级数据 (FXMinute)
- 外汇Tick数据 (TrueFX, Oanda)
- 历史数据文件 (HistData)

使用抽象工厂模式，支持运行时数据源切换和多源聚合。
"""

from .base import (
    AbstractDataSource,
    DataInterval,
    MarketType, 
    MarketData,
    DataSourceCapabilities
)

from .factory import (
    DataSourceFactory,
    DataSourceRegistry
)

from .converter import DataConverter

# 直接导入所有数据源实现
from .yfinance_source import YFinanceDataSource
from .truefx_source import TrueFXDataSource
from .oanda_source import OandaDataSource
from .histdata_source import HistDataDataSource
from .fxminute_source import FXMinuteDataSource


def get_available_sources():
    """
    获取当前环境中可用的数据源列表

    Returns:
        List[str]: 可用数据源名称列表
    """
    return ['yfinance', 'truefx', 'oanda', 'histdata', 'fxminute']


def create_source(source_type, config=None):
    """
    便捷函数：创建数据源实例

    Args:
        source_type (str): 数据源类型
        config (dict, optional): 配置参数

    Returns:
        AbstractDataSource: 数据源实例
    """
    return DataSourceFactory.create_data_source(source_type, config)



# 版本信息
__version__ = "1.0.0"
__author__ = "yuan"

# 导出的主要类
__all__ = [
    # 核心接口
    'AbstractDataSource',
    'DataInterval', 
    'MarketType',
    'MarketData',
    'DataSourceCapabilities',
    
    # 工厂模式
    'DataSourceFactory',
    'DataSourceRegistry',

    # 数据转换
    'DataConverter',
    
    # 数据源实现
    'YFinanceDataSource',
    'TrueFXDataSource',
    'OandaDataSource', 
    'HistDataDataSource',
    'FXMinuteDataSource',
    
    # 便捷函数
    'get_available_sources',
    'create_source'
]

