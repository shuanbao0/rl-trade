"""
数据管理模块
负责多数据源的数据获取、验证、缓存等功能

主要功能:
1. 多数据源支持 (YFinance, TrueFX, Oanda, HistData, FXMinute)
2. 数据验证和清洗  
3. 本地缓存管理
4. 异常处理和重试机制
5. 性能监控和日志记录

支持的数据源:
- YFinance: 免费股票数据 (实时+历史)
- TrueFX: 免费外汇数据 (实时+历史)
- Oanda: 专业外汇/CFD数据 (需要API密钥)
- HistData: 历史外汇文件数据 (手动下载)
- FXMinute: 本地FX-1-Minute-Data缓存数据 (data_cache/FX-1-Minute-Data)
"""

import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from dataclasses import dataclass
import re

from ..utils.logger import setup_logger, get_default_log_file
from ..utils.config import Config
from .sources import DataSourceFactory, AbstractDataSource
from .sources.base import DataInterval


@dataclass
class DataValidationResult:
    """数据验证结果"""
    is_valid: bool
    issues: List[str]
    records_count: int
    missing_values: int
    date_range: Tuple[str, str]


class DataManager:
    """
    数据管理器
    
    负责多数据源的数据获取、验证、缓存和管理
    采用单例模式确保数据一致性
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Optional[Config] = None, data_source_type: str = 'yfinance', data_source_config: Optional[Dict] = None):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[Config] = None, data_source_type: str = 'yfinance', data_source_config: Optional[Dict] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
            data_source_type: 数据源类型 (yfinance, truefx, oanda, histdata, fxminute)
            data_source_config: 数据源特定配置
        """
        if self._initialized:
            return
            
        self.config = config or Config()
        self.data_source_type = data_source_type
        self.data_source_config = data_source_config or {}
        
        # 初始化日志
        self.logger = setup_logger(
            name="DataManager",
            level="INFO",
            log_file=get_default_log_file("data_manager")
        )
        
        # 缓存相关
        self.cache_dir = self.config.data.cache_dir
        self.cache_expiry_hours = self.config.data.cache_expiry_hours
        self._cache = {}  # 内存缓存
        
        # 数据验证规则
        self.validation_rules = {
            'min_records': 10,  # 最少记录数
            'max_missing_ratio': 0.05,  # 最大缺失值比例 5%
            'price_min': 0.01,  # 最小价格
            'price_max': 100000,  # 最大价格
            'volume_min': 0,  # 最小成交量
        }
        
        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: {self.cache_dir}")
        
        # 初始化数据源
        self._init_data_source()
        
        self.logger.info(f"DataManager initialized successfully with {data_source_type} data source")
        self._initialized = True
    
    def _init_data_source(self):
        """
        初始化数据源
        """
        try:
            # 合并代理配置到数据源配置中
            final_config = self.data_source_config.copy()
            
            # 如果是 yfinance 数据源，添加代理支持
            if self.data_source_type == 'yfinance':
                # 从环境变量获取代理配置
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
                
            self.logger.info(f"Data source '{self.data_source_type}' initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data source '{self.data_source_type}': {e}")
            raise
    
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
        if self.data_source_type == 'yfinance':
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
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
        auto_adjust_interval: bool = False
    ) -> pd.DataFrame:
        """
        获取市场数据 (股票/外汇/其他)
        
        Args:
            symbol: 证券代码 (如: AAPL, EURUSD)
            period: 时间周期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: 是否强制刷新缓存
            auto_adjust_interval: 是否自动调整间隔以适应期间限制 (默认False)
            
        Returns:
            包含市场数据的DataFrame
            
        Raises:
            ValueError: 当参数无效时
            ConnectionError: 当网络连接失败时
            Exception: 当数据获取失败时
        """
        start_time = time.time()
        self.logger.info(f"Fetching data for {symbol}, period: {period}, interval: {interval}")
        
        try:
            # 参数验证
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            symbol = symbol.upper().strip()
            
            # 检查缓存
            cache_key = f"{self.data_source_type}_{symbol}_{period}_{interval}"
            
            if not force_refresh:
                cached_data = self._get_cached_data(cache_key)
                if cached_data is not None:
                    self.logger.info(f"Using cached data for {cache_key}")
                    return cached_data
            
            # 使用数据源接口获取数据
            data = self._fetch_from_data_source(symbol, period, interval)
            
            if data is None or data.empty:
                raise Exception(f"No data retrieved for symbol {symbol}")
            
            # 数据验证
            validation_result = self.validate_data(data)
            if not validation_result.is_valid:
                self.logger.warning(
                    f"Data validation issues for {symbol}: {validation_result.issues}"
                )
                # 如果问题严重，抛出异常
                if validation_result.records_count < self.validation_rules['min_records']:
                    raise Exception(f"Insufficient data records: {validation_result.records_count}")
            
            # 数据清洗
            cleaned_data = self._clean_data(data)
            
            # 缓存数据
            self._cache_data(cache_key, cleaned_data)
            
            duration = time.time() - start_time
            self.logger.info(
                f"Successfully fetched {len(cleaned_data)} records for {symbol} "
                f"in {duration:.2f} seconds"
            )
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
            raise
    
    def _fetch_from_data_source(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        从数据源获取数据，带重试机制
        
        Args:
            symbol: 证券代码
            period: 时间周期
            interval: 数据间隔
            
        Returns:
            原始数据DataFrame
        """
        max_retries = self.config.data.max_retries
        retry_delay = self.config.data.retry_delay
        
        # 计算时间范围
        # 对于FXMinute数据源，使用历史数据的结束时间
        if self.data_source_type == 'fxminute':
            end_date = datetime(2024, 12, 31, 23, 59, 59)  # FX-1-Minute-Data数据到2024年12月
        else:
            end_date = datetime.now()
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
                    if "too many requests" in error_str or "rate limited" in error_str:
                        delay = min(retry_delay * (4 ** attempt), 120)
                        self.logger.info(f"Rate limited, waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                    elif "timeout" in error_str or "connection" in error_str:
                        delay = retry_delay * (2 ** attempt)
                        self.logger.info(f"Connection issue, waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
                    else:
                        delay = retry_delay * (2 ** attempt)
                        time.sleep(delay)
                else:
                    self.logger.error(f"All data source fetch attempts failed for {symbol}")
                    raise ConnectionError(f"Failed to fetch data after {max_retries} attempts")
        
        return None
    
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
            '5d': DataInterval.DAY_1,  # 映射到DAY_1，因为FXMinute只支持1分钟数据
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
        period = period.lower()
        
        # 使用正则表达式解析
        match = re.match(r'(\d+)([a-z]+)', period)
        if not match:
            # 特殊情况
            if period == 'ytd':
                return 365  # 约一年
            elif period == 'max':
                return 25 * 365  # 25年，适合FX-1-Minute-Data的完整历史数据
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
    
    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """
        验证数据质量
        
        Args:
            data: 要验证的数据
            
        Returns:
            数据验证结果
        """
        issues = []
        
        # 基本检查
        if data is None or data.empty:
            return DataValidationResult(
                is_valid=False,
                issues=["Data is None or empty"],
                records_count=0,
                missing_values=0,
                date_range=("", "")
            )
        
        records_count = len(data)
        missing_values = data.isnull().sum().sum()
        
        # 检查记录数量
        if records_count < self.validation_rules['min_records']:
            issues.append(f"Insufficient records: {records_count} < {self.validation_rules['min_records']}")
        
        # 检查缺失值比例
        missing_ratio = missing_values / (records_count * len(data.columns))
        if missing_ratio > self.validation_rules['max_missing_ratio']:
            issues.append(f"Too many missing values: {missing_ratio:.2%} > {self.validation_rules['max_missing_ratio']:.2%}")
        
        # 检查时间连续性
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = (
                data.index.min().strftime('%Y-%m-%d'),
                data.index.max().strftime('%Y-%m-%d')
            )
            
            # 检查重复日期
            duplicate_dates = data.index.duplicated()
            if duplicate_dates.any():
                issues.append(f"Duplicate dates found: {duplicate_dates.sum()} records")
        else:
            date_range = ("", "")
            issues.append("Index is not DatetimeIndex")
        
        is_valid = len(issues) == 0
        
        return DataValidationResult(
            is_valid=is_valid,
            issues=issues,
            records_count=records_count,
            missing_values=missing_values,
            date_range=date_range
        )
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = data.copy()
        
        # 删除重复行
        original_count = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        dropped_duplicates = original_count - len(cleaned_data)
        
        if dropped_duplicates > 0:
            self.logger.info(f"Removed {dropped_duplicates} duplicate records")
        
        # 处理缺失值 - 使用前向填充
        missing_before = cleaned_data.isnull().sum().sum()
        cleaned_data = cleaned_data.ffill()
        
        # 如果还有缺失值，使用后向填充
        cleaned_data = cleaned_data.bfill()
        
        missing_after = cleaned_data.isnull().sum().sum()
        
        if missing_before > missing_after:
            self.logger.info(f"Filled {missing_before - missing_after} missing values")
        
        # 确保数据类型正确
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        
        # 排序确保时间顺序
        if isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data = cleaned_data.sort_index()
        
        return cleaned_data
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        # 检查内存缓存
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                return cache_entry['data']
            else:
                del self._cache[cache_key]
        
        # 检查文件缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                # 检查文件修改时间
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if self._is_cache_valid(file_time):
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # 更新内存缓存
                    self._cache[cache_key] = {
                        'data': data,
                        'timestamp': file_time
                    }
                    
                    return data
                else:
                    # 删除过期文件
                    os.remove(cache_file)
                    self.logger.debug(f"Removed expired cache file: {cache_file}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
                try:
                    os.remove(cache_file)
                except:
                    pass
        
        return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        缓存数据到内存和文件
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        timestamp = datetime.now()
        
        # 内存缓存
        self._cache[cache_key] = {
            'data': data.copy(),
            'timestamp': timestamp
        }
        
        # 文件缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.debug(f"Cached data to file: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache data to file {cache_file}: {str(e)}")
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """
        检查缓存是否有效
        
        Args:
            timestamp: 缓存时间戳
            
        Returns:
            是否有效
        """
        expiry_time = timestamp + timedelta(hours=self.cache_expiry_hours)
        return datetime.now() < expiry_time
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        清理缓存
        
        Args:
            symbol: 股票代码，如果为None则清理所有缓存
        """
        if symbol:
            symbol = symbol.upper()
            # 清理内存缓存
            keys_to_remove = [key for key in self._cache.keys() if symbol in key]
            for key in keys_to_remove:
                del self._cache[key]
            
            # 清理文件缓存
            cache_files = [f for f in os.listdir(self.cache_dir) if symbol in f and f.endswith('.pkl')]
            for file in cache_files:
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {file}: {str(e)}")
            
            self.logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            # 清理所有缓存
            self._cache.clear()
            
            try:
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
            except Exception as e:
                self.logger.warning(f"Failed to clear all cache files: {str(e)}")
            
            self.logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        memory_cache_count = len(self._cache)
        
        file_cache_count = 0
        total_cache_size = 0
        
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            file_cache_count = len(cache_files)
            
            for file in cache_files:
                file_path = os.path.join(self.cache_dir, file)
                total_cache_size += os.path.getsize(file_path)
                
        except Exception as e:
            self.logger.warning(f"Failed to get cache info: {str(e)}")
        
        return {
            'memory_cache_count': memory_cache_count,
            'file_cache_count': file_cache_count,
            'total_cache_size_mb': total_cache_size / (1024 * 1024),
            'cache_directory': self.cache_dir,
            'cache_expiry_hours': self.cache_expiry_hours
        }
    
    def get_data_source_info(self) -> Dict[str, any]:
        """
        获取数据源信息
        
        Returns:
            数据源信息字典
        """
        capabilities = self.data_source.get_capabilities()
        return {
            'data_source_type': self.data_source_type,
            'data_source_name': capabilities.name,
            'supported_markets': [str(market) for market in capabilities.supported_markets],
            'supported_intervals': [str(interval) for interval in capabilities.supported_intervals],
            'has_realtime': capabilities.has_realtime,
            'has_historical': capabilities.has_historical,
            'config': self.data_source_config
        }