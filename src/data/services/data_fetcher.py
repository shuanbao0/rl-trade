"""
数据获取服务
负责核心数据获取逻辑，包括数据源调用、重试机制等
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from ..sources.base import DataSource, DataInterval, MarketType
from ...utils.config import Config
from ...utils.logger import get_logger


class DataFetcher:
    """数据获取服务"""
    
    def __init__(self, config: Config, data_source, logger=None):
        """
        初始化数据获取服务
        
        Args:
            config: 配置对象
            data_source: 数据源实例
            logger: 日志器
        """
        self.config = config
        self.data_source = data_source
        self.logger = logger or get_logger(__name__)
    
    def fetch_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        获取数据的核心方法
        
        Args:
            symbol: 证券代码
            period: 时间周期
            interval: 数据间隔
            
        Returns:
            数据DataFrame或None
        """
        max_retries = self.config.data.max_retries
        retry_delay = self.config.data.retry_delay
        
        # 计算时间范围
        end_date = self._get_end_date()
        start_date = self._calculate_start_date(period, end_date)
        
        # 转换间隔格式
        data_interval = self._convert_interval(interval)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Data source fetch attempt {attempt + 1} for {symbol}")
                
                # 使用数据源接口获取数据
                data = self.data_source.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=data_interval
                )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"Successfully fetched {len(data)} records from data source")
                    return data
                else:
                    self.logger.warning(f"Empty data returned for {symbol}")
                    
            except Exception as e:
                error_str = str(e).lower()
                self.logger.warning(
                    f"Data source fetch attempt {attempt + 1} failed for {symbol}: {str(e)}"
                )
                
                if attempt < max_retries - 1:
                    # 针对不同错误类型使用不同的重试策略
                    delay = self._calculate_retry_delay(error_str, retry_delay, attempt)
                    self.logger.info(f"Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All data source fetch attempts failed for {symbol}")
                    raise ConnectionError(f"Failed to fetch data after {max_retries} attempts")
        
        return None
    
    def fetch_data_by_date_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        按日期范围获取数据
        
        Args:
            symbol: 证券代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            数据DataFrame或None
        """
        max_retries = self.config.data.max_retries
        retry_delay = self.config.data.retry_delay
        
        # 转换间隔格式
        data_interval = self._convert_interval(interval)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Data source fetch attempt {attempt + 1} for {symbol} in date range")
                
                # 使用数据源接口获取数据
                data = self.data_source.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=data_interval
                )
                
                if data is not None and not data.empty:
                    self.logger.debug(f"Successfully fetched {len(data)} records from data source")
                    return data
                else:
                    self.logger.warning(f"Empty data returned for {symbol} in date range")
                    
            except Exception as e:
                error_str = str(e).lower()
                self.logger.warning(
                    f"Data source fetch attempt {attempt + 1} failed for {symbol}: {str(e)}"
                )
                
                if attempt < max_retries - 1:
                    delay = self._calculate_retry_delay(error_str, retry_delay, attempt)
                    self.logger.info(f"Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All data source fetch attempts failed for {symbol} in date range")
                    raise ConnectionError(f"Failed to fetch data after {max_retries} attempts")
        
        return None
    
    def fetch_from_source(
        self,
        source_name: str,
        symbol: str,
        period: str,
        interval: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        从指定数据源获取数据
        
        Args:
            source_name: 数据源名称
            symbol: 证券代码
            period: 时间周期
            interval: 数据间隔
            **kwargs: 其他参数
            
        Returns:
            数据DataFrame
        """
        from ..sources import DataSourceFactory
        
        # 创建数据源实例
        data_source = DataSourceFactory.create_data_source(source_name)
        
        try:
            # 连接数据源
            data_source.connect()
            
            # 计算时间范围
            end_date = self._get_end_date()
            start_date = self._calculate_start_date(period, end_date)
            
            # 转换间隔格式
            data_interval = self._convert_interval(interval)
            
            # 获取数据
            data = data_source.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=data_interval
            )
            
            return data
            
        finally:
            # 断开连接
            try:
                data_source.disconnect()
            except:
                pass
    
    def _get_end_date(self) -> datetime:
        """获取结束日期"""
        # 对于FXMinute数据源，使用历史数据的结束时间
        if hasattr(self, 'data_source_type') and self.data_source_type == DataSource.FXMINUTE:
            return datetime(2024, 12, 31, 23, 59, 59)
        else:
            return datetime.now()
    
    def _calculate_start_date(self, period: str, end_date: datetime) -> datetime:
        """
        根据周期计算开始日期
        
        Args:
            period: 时间周期字符串
            end_date: 结束日期
            
        Returns:
            开始日期
        """
        period_days = self._period_to_days(period)
        return end_date - timedelta(days=period_days)
    
    def _convert_interval(self, interval: str) -> DataInterval:
        """
        转换间隔格式
        
        Args:
            interval: 间隔字符串 (1m, 1h, 1d 等)
            
        Returns:
            DataInterval 枚举
        """
        interval_map = {
            '1m': DataInterval.MINUTE_1,
            '2m': DataInterval.MINUTE_2,
            '5m': DataInterval.MINUTE_5,
            '15m': DataInterval.MINUTE_15,
            '30m': DataInterval.MINUTE_30,
            '1h': DataInterval.HOUR_1,
            '60m': DataInterval.HOUR_1,
            '90m': DataInterval.MINUTE_90,
            '1d': DataInterval.DAY_1,
            '3d': DataInterval.DAY_3,
            '5d': DataInterval.DAY_1,
            '1wk': DataInterval.WEEK_1,
            '1mo': DataInterval.MONTH_1,
            '3mo': DataInterval.MONTH_3
        }
        
        return interval_map.get(interval, DataInterval.DAY_1)
    
    def _period_to_days(self, period: str) -> int:
        """
        将period字符串转换为天数
        
        Args:
            period: 时间周期字符串
            
        Returns:
            天数
        """
        import re
        
        period = period.lower()
        
        # 使用正则表达式解析
        match = re.match(r'(\d+)([a-z]+)', period)
        if not match:
            # 特殊情况
            if period == 'ytd':
                return 365
            elif period == 'max':
                return 25 * 365  # 25年
            else:
                return 365  # 默认一年
        
        num, unit = match.groups()
        num = int(num)
        
        unit_multipliers = {
            'd': 1,       # 天
            'w': 7,       # 周
            'mo': 30,     # 月
            'm': 30,      # 月的另一种表示
            'y': 365,     # 年
        }
        
        return num * unit_multipliers.get(unit, 1)
    
    def _calculate_retry_delay(self, error_str: str, base_delay: float, attempt: int) -> float:
        """
        计算重试延迟时间
        
        Args:
            error_str: 错误信息字符串（小写）
            base_delay: 基础延迟时间
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            延迟时间（秒）
        """
        if "too many requests" in error_str or "rate limited" in error_str:
            return min(base_delay * (4 ** attempt), 120)
        elif "timeout" in error_str or "connection" in error_str:
            return base_delay * (2 ** attempt)
        else:
            return base_delay * (2 ** attempt)