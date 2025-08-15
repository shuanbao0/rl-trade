"""
重构后的数据管理模块
采用服务化架构，将功能分散到专门的服务类中
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd

from ...utils.logger import setup_logger, get_default_log_file
from ...utils.config import Config
from ..sources import DataSourceFactory
from ..sources.base import DataSource, MarketType, DataPeriod
from ..managers.batch_downloader import BatchDownloader
from ..processors.market_detector import MarketTypeDetector
from ..managers.routing_manager import get_routing_manager
from ..advisors.compatibility_checker import get_compatibility_checker
from ..processors.market_processors import MarketProcessorFactory
from ..managers.cache_manager import get_cache_manager, CacheKey
from .exceptions import (
    DataSourceError,
    DataInsufficientError,
    AllDataSourcesFailedError
)
from ...utils.date_range_utils import DateRange, DateRangeUtils

# 导入服务类
from ..services import DataFetcher, DataValidator, DataValidationResult, DateRangeFetcher


class DataManager:
    """
    重构后的数据管理器
    
    采用服务化架构，将各个功能委托给专门的服务类
    保持原有的公共接口不变，确保向后兼容性
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[Config] = None, data_source_type: Union[str, DataSource] = 'yfinance', data_source_config: Optional[Dict] = None):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Config] = None, data_source_type: Union[str, DataSource] = 'yfinance', data_source_config: Optional[Dict] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
            data_source_type: 数据源类型 (可以是字符串或DataSource枚举)
            data_source_config: 数据源特定配置
        """
        if self._initialized:
            return
            
        self.config = config or Config()
        
        # 转换数据源类型为枚举
        if isinstance(data_source_type, str):
            self.data_source_type = DataSource.from_string(data_source_type)
        else:
            self.data_source_type = data_source_type
            
        self.data_source_config = data_source_config or {}
        
        # 初始化日志
        self.logger = setup_logger(
            name="DataManager",
            level="INFO",
            log_file=get_default_log_file("data_manager")
        )
        
        # 确保缓存目录存在
        cache_dir = self.config.data.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            self.logger.info(f"Created cache directory: {cache_dir}")
        
        # 初始化核心组件
        self._init_core_components()
        
        # 初始化数据源
        self._init_data_source()
        
        # 初始化服务层
        self._init_services()
        
        self.logger.info(f"DataManager initialized successfully with {self.data_source_type.value} data source")
        self._initialized = True
    
    def _init_core_components(self):
        """初始化核心组件"""
        # 市场类型检测器
        self.market_detector = MarketTypeDetector()
        
        # 路由管理器
        self.routing_manager = get_routing_manager()
        
        # 兼容性检查器
        self.compatibility_checker = get_compatibility_checker()
        
        # 缓存管理器
        self.cache_manager = get_cache_manager()
        
        # 批次下载器
        self.batch_downloader = BatchDownloader(config=self.config, data_manager=self)
    
    def _init_data_source(self):
        """初始化数据源"""
        try:
            # 合并代理配置到数据源配置中
            final_config = self.data_source_config.copy()
            
            # 如果是 yfinance 数据源，添加代理支持
            if self.data_source_type == DataSource.YFINANCE:
                use_proxy = os.getenv('USE_PROXY', 'false').lower() == 'true'
                if use_proxy:
                    proxy_host = os.getenv('PROXY_HOST', '127.0.0.1')
                    proxy_port = os.getenv('PROXY_PORT', '7891')
                    final_config['proxy'] = f"http://{proxy_host}:{proxy_port}"
            
            # 创建数据源实例
            self.data_source = DataSourceFactory.create_data_source(
                self.data_source_type, final_config
            )
            
            # 连接数据源
            if hasattr(self.data_source, 'connect'):
                self.data_source.connect()
                
            self.logger.info(f"Data source '{self.data_source_type.value}' initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data source '{self.data_source_type.value}': {e}")
            raise
    
    def _init_services(self):
        """初始化服务层"""
        # 数据获取服务
        self.data_fetcher = DataFetcher(
            config=self.config,
            data_source=self.data_source,
            logger=self.logger
        )
        self.data_fetcher.data_source_type = self.data_source_type  # 传递数据源类型
        
        # 数据验证服务
        validation_rules = {
            'min_records': 10,
            'max_missing_ratio': 0.05,
            'price_min': 0.01,
            'price_max': 100000,
            'volume_min': 0,
        }
        self.data_validator = DataValidator(
            validation_rules=validation_rules,
            logger=self.logger
        )
        
        # 日期范围数据获取服务
        self.date_range_fetcher = DateRangeFetcher(
            config=self.config,
            data_fetcher=self.data_fetcher,
            cache_manager=self.cache_manager,
            batch_downloader=self.batch_downloader,
            market_detector=self.market_detector,
            logger=self.logger
        )
    
    def set_proxy(self, proxy_url: str):
        """
        设置代理配置
        
        Args:
            proxy_url: 代理URL (例如: socks5://127.0.0.1:7891)
        """
        self.logger.info(f"代理配置已更新: {proxy_url}")
        
        # 更新数据源配置中的代理
        if hasattr(self.data_source, 'update_config'):
            self.data_source.update_config({'proxy': proxy_url})
        
        # 对于 yfinance 数据源，还需要设置全局代理
        if self.data_source_type == DataSource.YFINANCE:
            try:
                import yfinance as yf
                if hasattr(yf, 'set_config'):
                    yf.set_config(proxy=proxy_url)
                    self.logger.info("yfinance 全局代理配置已设置")
                else:
                    # 设置环境变量
                    os.environ['HTTP_PROXY'] = proxy_url
                    os.environ['HTTPS_PROXY'] = proxy_url
                    os.environ['http_proxy'] = proxy_url
                    os.environ['https_proxy'] = proxy_url
                    self.logger.info("通过环境变量设置代理配置")
            except ImportError:
                self.logger.warning("yfinance not available for proxy configuration")
    
    def get_stock_data(
        self,
        symbol: str,
        period: Union[str, DataPeriod] = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
        auto_adjust_interval: bool = False
    ) -> pd.DataFrame:
        """
        获取市场数据 (股票/外汇/其他)
        
        Args:
            symbol: 证券代码 (如: AAPL, EURUSD)
            period: 时间周期 - 支持字符串或DataPeriod枚举
            interval: 数据间隔
            force_refresh: 是否强制刷新缓存
            auto_adjust_interval: 是否自动调整间隔以适应期间限制
            
        Returns:
            包含市场数据的DataFrame
        """
        start_time = time.time()
        
        # 转换DataPeriod枚举为字符串
        period_str = period.value if isinstance(period, DataPeriod) else str(period)
        
        self.logger.info(f"Fetching data for {symbol}, period: {period_str}, interval: {interval}")
        
        try:
            # 参数验证
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            symbol = symbol.upper().strip()
            
            # 标准化period参数
            if isinstance(period, DataPeriod):
                period_str = period.value
            else:
                try:
                    period_enum = DataPeriod.from_string(str(period))
                    period_str = period_enum.value
                except ValueError:
                    period_str = str(period)
                    self.logger.warning(f"Using non-standard period format: {period_str}")
            
            # 自动检测市场类型
            market_type = self.market_detector.detect(symbol)
            
            # 检查缓存
            cache_key = CacheKey(
                source=self.data_source_type.value,
                symbol=symbol,
                market_type=market_type,
                period=period_str,
                interval=interval
            )
            
            if not force_refresh:
                cached_data = self.cache_manager.get(cache_key)
                if cached_data is not None:
                    self.logger.info(f"Using cached data for {cache_key.to_string()}")
                    return cached_data
            
            # 获取数据（智能判断是否使用分批次下载）
            data = self._smart_fetch_data(symbol, period_str, interval)
            
            if data is None or data.empty:
                raise DataSourceError(f"No data retrieved for symbol {symbol}", symbol=symbol)
            
            # 数据验证
            validation_result = self.data_validator.validate_data(data)
            if not validation_result.is_valid:
                self.logger.warning(f"Data validation issues for {symbol}: {validation_result.issues}")
                if validation_result.records_count < 10:  # min_records
                    raise DataInsufficientError(
                        f"Insufficient data records: {validation_result.records_count}",
                        expected_records=10,
                        actual_records=validation_result.records_count,
                        symbol=symbol
                    )
            
            # 数据清洗
            cleaned_data = self.data_validator.clean_data(data)
            
            # 缓存数据
            self.cache_manager.put(cache_key, cleaned_data)
            
            duration = time.time() - start_time
            self.logger.info(
                f"Successfully fetched {len(cleaned_data)} records for {symbol} "
                f"in {duration:.2f} seconds"
            )
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            raise
    
    def get_stock_data_by_date_range(
        self,
        symbol: str,
        start_date: Union[str, datetime, None] = None,
        end_date: Union[str, datetime, None] = None,
        period: Union[str, DataPeriod, None] = None,
        interval: str = "1d",
        force_refresh: bool = False,
        auto_adjust_interval: bool = False
    ) -> pd.DataFrame:
        """
        根据日期范围获取市场数据
        
        Args:
            symbol: 证券代码
            start_date: 开始日期
            end_date: 结束日期
            period: 时间周期
            interval: 数据间隔
            force_refresh: 是否强制刷新缓存
            auto_adjust_interval: 是否自动调整间隔
            
        Returns:
            包含市场数据的DataFrame
        """
        # 委托给日期范围获取服务
        data = self.date_range_fetcher.fetch_data_by_date_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            force_refresh=force_refresh,
            data_source_type=self.data_source_type
        )
        
        # 数据验证
        validation_result = self.data_validator.validate_data(data)
        if not validation_result.is_valid:
            self.logger.warning(f"Data validation issues for {symbol}: {validation_result.issues}")
            if validation_result.records_count < 10:
                raise DataInsufficientError(
                    f"Insufficient data records: {validation_result.records_count}",
                    expected_records=10,
                    actual_records=validation_result.records_count,
                    symbol=symbol
                )
        
        # 数据清洗
        cleaned_data = self.data_validator.clean_data(data)
        
        return cleaned_data
    
    def _smart_fetch_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        智能数据获取：自动判断是否使用分批次下载
        
        Args:
            symbol: 交易符号
            period: 数据周期
            interval: 数据间隔
            
        Returns:
            数据DataFrame
        """
        # 判断是否需要分批次下载
        should_use_batch = self._should_use_batch_download(period, interval)
        
        if should_use_batch:
            self.logger.info(f"使用分批次下载模式获取 {symbol} 数据")
            return self._fetch_with_batch_download(symbol, period, interval)
        else:
            self.logger.info(f"使用常规下载模式获取 {symbol} 数据")
            return self.data_fetcher.fetch_data(symbol, period, interval)
    
    def _should_use_batch_download(self, period: str, interval: str) -> bool:
        """判断是否应该使用分批次下载"""
        period_days = self._parse_period_to_days(period)
        estimated_records = self._estimate_record_count(period_days, interval)
        
        # 从配置获取阈值
        auto_threshold_days = getattr(self.config.data, 'auto_batch_threshold_days', 365)
        
        conditions = [
            period_days > auto_threshold_days,
            estimated_records > 10000,
            (interval in ['1m', '5m', '15m', '30m', '1h'] and period_days > 30)
        ]
        
        return any(conditions)
    
    def _fetch_with_batch_download(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """使用分批次下载获取数据"""
        # 转换period为日期范围
        date_range = DateRangeUtils.create_date_range(period=period)
        
        # 使用批次下载器
        return self.batch_downloader.download_in_batches(symbol, date_range, interval)
    
    def _parse_period_to_days(self, period: str) -> int:
        """解析周期字符串到天数"""
        if period == 'max':
            return 365 * 20  # 假设最大20年
        
        period = period.lower()
        
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('w'):
            return int(period[:-1]) * 7
        elif period.endswith('mo'):
            return int(period[:-2]) * 30
        elif period.endswith('y'):
            return int(period[:-1]) * 365
        else:
            try:
                return int(period)
            except:
                return 365
    
    def _estimate_record_count(self, days: int, interval: str) -> int:
        """估算记录数量"""
        interval_multipliers = {
            '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
            '1h': 24, '1d': 1, '1wk': 0.14, '1mo': 0.03,
        }
        multiplier = interval_multipliers.get(interval, 1)
        return int(days * multiplier)
    
    # 委托方法
    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """验证数据质量（委托给验证服务）"""
        return self.data_validator.validate_data(data)
    
    def get_date_range_estimation(
        self,
        symbol: str,
        start_date: Union[str, datetime, None] = None,
        end_date: Union[str, datetime, None] = None,
        period: Union[str, DataPeriod, None] = None,
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """获取日期范围下载估算（委托给日期范围服务）"""
        return self.date_range_fetcher.get_date_range_estimation(
            symbol, start_date, end_date, period, interval, self.data_source_type
        )
    
    def convert_period_to_date_range(
        self,
        period: Union[str, DataPeriod],
        end_date: Optional[Union[str, datetime]] = None
    ) -> DateRange:
        """将数据周期转换为日期范围（委托给日期范围服务）"""
        return self.date_range_fetcher.convert_period_to_date_range(period, end_date)
    
    # 保持向后兼容的方法
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """清理缓存"""
        if symbol:
            # 实现符号特定的缓存清理
            self.cache_manager.clear_by_symbol(symbol.upper())
            self.logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            self.cache_manager.clear_all()
            self.logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return self.cache_manager.get_statistics()
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        capabilities = self.data_source.get_capabilities()
        return {
            'data_source_type': self.data_source_type.value,
            'data_source_name': capabilities.name,
            'supported_markets': [str(market) for market in capabilities.supported_markets],
            'supported_intervals': [str(interval) for interval in capabilities.supported_intervals],
            'has_realtime': capabilities.has_realtime,
            'has_historical': capabilities.has_historical,
            'config': self.data_source_config
        }
    
    # DataPeriod相关方法
    def get_supported_periods(self) -> List[DataPeriod]:
        """获取支持的数据周期枚举列表"""
        return list(DataPeriod)
    
    def get_period_info(self, period: Union[str, DataPeriod]) -> Dict[str, Any]:
        """获取数据周期的详细信息"""
        try:
            if isinstance(period, str):
                period_enum = DataPeriod.from_string(period)
            else:
                period_enum = period
                
            return {
                'enum_value': period_enum,
                'string_value': period_enum.value,
                'display_name': period_enum.display_name,
                'days': period_enum.to_days(),
                'is_short_term': period_enum.is_short_term,
                'is_medium_term': period_enum.is_medium_term,
                'is_long_term': period_enum.is_long_term,
                'recommended_interval': period_enum.get_recommended_interval(self.data_source_type)
            }
        except ValueError as e:
            return {
                'error': str(e),
                'enum_value': None,
                'string_value': str(period),
                'supported_periods': [p.value for p in DataPeriod]
            }