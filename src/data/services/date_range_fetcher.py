"""
日期范围数据获取服务
专门处理基于日期范围的数据获取逻辑
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

from ..sources.base import MarketType
from ...utils.date_range_utils import DateRangeUtils, DateRange
from ...utils.config import Config
from ...utils.logger import get_logger
from ..managers.cache_manager import CacheKey
from .data_fetcher import DataFetcher


class DateRangeFetcher:
    """日期范围数据获取服务"""
    
    def __init__(
        self, 
        config: Config, 
        data_fetcher: DataFetcher,
        cache_manager,
        batch_downloader,
        market_detector,
        logger=None
    ):
        """
        初始化日期范围数据获取服务
        
        Args:
            config: 配置对象
            data_fetcher: 数据获取服务
            cache_manager: 缓存管理器
            batch_downloader: 批次下载器
            market_detector: 市场类型检测器
            logger: 日志器
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.cache_manager = cache_manager
        self.batch_downloader = batch_downloader
        self.market_detector = market_detector
        self.logger = logger or get_logger(__name__)
        
        # 日期范围工具
        self.date_utils = DateRangeUtils()
    
    def fetch_data_by_date_range(
        self,
        symbol: str,
        start_date: Union[str, datetime, None] = None,
        end_date: Union[str, datetime, None] = None,
        period: Union[str, None] = None,
        interval: str = "1d",
        force_refresh: bool = False,
        data_source_type = None
    ) -> pd.DataFrame:
        """
        根据日期范围获取数据
        
        Args:
            symbol: 证券代码
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期（如果未指定日期范围时使用）
            interval: 数据间隔
            force_refresh: 是否强制刷新缓存
            data_source_type: 数据源类型
            
        Returns:
            数据DataFrame
        """
        try:
            # 参数验证
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            symbol = symbol.upper().strip()
            
            # 创建日期范围
            try:
                date_range = self.date_utils.create_date_range(
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
            except ValueError as e:
                raise ValueError(f"Invalid date range parameters: {e}")
            
            self.logger.info(
                f"Fetching data for {symbol} from {date_range.start_date.date()} "
                f"to {date_range.end_date.date()} ({date_range.duration_days} days), interval: {interval}"
            )
            
            # 日期范围验证
            if data_source_type:
                is_valid, validation_errors = DateRangeUtils.validate_date_range(
                    date_range, data_source_type
                )
                
                if not is_valid:
                    raise ValueError(f"Date range validation failed: {', '.join(validation_errors)}")
            
            # 自动检测市场类型
            market_type = self.market_detector.detect(symbol)
            
            # 生成缓存键（使用日期范围标识）
            cache_key = CacheKey(
                source=data_source_type.value if data_source_type else "unknown",
                symbol=symbol,
                market_type=market_type,
                period=f"{date_range.start_date.date()}_{date_range.end_date.date()}",
                interval=interval
            )
            
            # 检查缓存
            if not force_refresh:
                cached_data = self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.logger.info(f"Using cached data for {cache_key.to_string()}")
                    return cached_data
            
            # 获取数据（使用日期范围）
            data = self._fetch_data_by_range(symbol, date_range, interval)
            
            if data is None or data.empty:
                from ..exceptions import DataSourceError
                raise DataSourceError(f"No data retrieved for symbol {symbol} in date range", symbol=symbol)
            
            # 缓存数据
            self.cache_manager.put(cache_key, data)
            
            self.logger.info(
                f"Successfully fetched {len(data)} records for {symbol} in date range"
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data by date range for {symbol}: {str(e)}")
            raise
    
    def _fetch_data_by_range(
        self,
        symbol: str,
        date_range: DateRange,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        根据日期范围获取数据
        
        Args:
            symbol: 交易代码
            date_range: 日期范围对象
            interval: 数据间隔
            
        Returns:
            数据 DataFrame
        """
        # 判断是否需要分批次下载
        should_use_batch = self._should_use_batch_download(date_range, interval)
        
        if should_use_batch:
            self.logger.info(f"使用分批次下载模式获取 {symbol} 数据")
            return self.batch_downloader.download_in_batches(symbol, date_range, interval)
        else:
            self.logger.info(f"使用常规下载模式获取 {symbol} 数据")
            return self.data_fetcher.fetch_data_by_date_range(
                symbol, date_range.start_date, date_range.end_date, interval
            )
    
    def _should_use_batch_download(self, date_range: DateRange, interval: str) -> bool:
        """
        判断是否应该对日期范围使用分批次下载
        
        Args:
            date_range: 日期范围对象
            interval: 数据间隔
            
        Returns:
            是否需要分批次下载
        """
        # 使用批次下载器的判断逻辑
        return self.batch_downloader.should_use_batch_download(
            date_range, interval
        )
    
    def get_date_range_estimation(
        self,
        symbol: str,
        start_date: Union[str, datetime, None] = None,
        end_date: Union[str, datetime, None] = None,
        period: Union[str, None] = None,
        interval: str = "1d",
        data_source_type = None
    ) -> Dict[str, Any]:
        """
        获取日期范围下载的时间估算和建议
        
        Args:
            symbol: 交易代码
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期
            interval: 数据间隔
            data_source_type: 数据源类型
            
        Returns:
            估算信息字典
        """
        try:
            # 创建日期范围
            date_range = self.date_utils.create_date_range(
                start_date=start_date,
                end_date=end_date,
                period=period
            )
            
            # 验证日期范围
            is_valid = True
            validation_errors = []
            if data_source_type:
                is_valid, validation_errors = DateRangeUtils.validate_date_range(
                    date_range, data_source_type
                )
            
            # 获取推荐间隔
            recommended_interval = "1d"  # 默认推荐
            if data_source_type:
                try:
                    recommended_interval = DateRangeUtils.suggest_optimal_interval(
                        date_range, data_source_type
                    )
                except:
                    pass
            
            # 获取批次下载估算
            batch_estimation = {}
            if hasattr(self.batch_downloader, 'get_batch_download_estimation_for_range'):
                try:
                    batch_estimation = self.batch_downloader.get_batch_download_estimation_for_range(
                        symbol, date_range, interval
                    )
                except:
                    pass
            
            return {
                'date_range': date_range.to_dict(),
                'validation': {
                    'is_valid': is_valid,
                    'errors': validation_errors
                },
                'recommendations': {
                    'optimal_interval': recommended_interval,
                    'current_interval': interval,
                    'interval_suggestion': (
                        f"Consider using '{recommended_interval}' for better performance"
                        if recommended_interval != interval else "Current interval is optimal"
                    )
                },
                'estimation': batch_estimation,
                'batch_download': {
                    'recommended': self._should_use_batch_download(date_range, interval),
                    'reason': self._get_batch_download_reason(date_range, interval)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'date_range': None,
                'validation': {'is_valid': False, 'errors': [str(e)]},
                'recommendations': {},
                'estimation': {},
                'batch_download': {'recommended': False, 'reason': 'Error in analysis'}
            }
    
    def _get_batch_download_reason(self, date_range: DateRange, interval: str) -> str:
        """
        获取分批次下载推荐的原因
        
        Args:
            date_range: 日期范围
            interval: 数据间隔
            
        Returns:
            推荐原因说明
        """
        period_days = date_range.duration_days
        estimated_records = self._estimate_record_count(period_days, interval)
        
        # 从配置获取阈值
        auto_threshold_days = getattr(self.config.data, 'auto_batch_threshold_days', 365)
        
        reasons = []
        
        if period_days > auto_threshold_days:
            reasons.append(f"Long time span ({period_days} days > {auto_threshold_days} threshold)")
            
        if estimated_records > 10000:
            reasons.append(f"Large dataset ({estimated_records:,} estimated records > 10,000 threshold)")
            
        high_freq_intervals = ['1m', '5m', '15m', '30m', '1h']
        if (interval in high_freq_intervals and period_days > 30):
            reasons.append(f"High frequency data ({interval}) over 30 days")
        
        if reasons:
            return "; ".join(reasons)
        else:
            return "Small dataset, regular download is sufficient"
    
    def _estimate_record_count(self, days: int, interval: str) -> int:
        """
        估算记录数量
        
        Args:
            days: 天数
            interval: 时间间隔
            
        Returns:
            估算的记录数
        """
        interval_multipliers = {
            '1m': 1440,    # 每天1440条记录
            '5m': 288,     # 每天288条记录
            '15m': 96,     # 每天96条记录
            '30m': 48,     # 每天48条记录
            '1h': 24,      # 每天24条记录
            '1d': 1,       # 每天1条记录
            '1wk': 0.14,   # 每周1条记录
            '1mo': 0.03,   # 每月1条记录
        }
        
        multiplier = interval_multipliers.get(interval, 1)
        return int(days * multiplier)
    
    def convert_period_to_date_range(
        self,
        period: Union[str, None],
        end_date: Optional[Union[str, datetime]] = None
    ) -> DateRange:
        """
        将数据周期转换为日期范围
        
        Args:
            period: 数据周期
            end_date: 结束日期，默认为当前日期
            
        Returns:
            日期范围对象
        """
        return self.date_utils.create_date_range(
            start_date=None,
            end_date=end_date,
            period=period
        )