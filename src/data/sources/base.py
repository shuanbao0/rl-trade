"""
数据源抽象基类和核心数据结构定义
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging


class DataInterval(Enum):
    """数据间隔枚举"""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"  
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_2 = "2m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    MINUTE_90 = "90m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    YEAR_1 = "1Y"


class MarketType(Enum):
    """市场类型枚举"""
    STOCK = "stock"
    FOREX = "forex"  
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"
    COMMODITIES = "commodities"
    BONDS = "bonds"
    INDEX = "index"
    ETF = "etf"


class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"        # 高质量：官方API，实时更新
    MEDIUM = "medium"    # 中等质量：延迟较小，数据完整
    LOW = "low"         # 低质量：延迟较大，可能有缺失
    UNKNOWN = "unknown"  # 未知质量


@dataclass
class MarketData:
    """统一的市场数据格式"""
    symbol: str                           # 标的代码
    timestamp: datetime                   # 时间戳
    open: float                          # 开盘价
    high: float                          # 最高价
    low: float                           # 最低价
    close: float                         # 收盘价
    volume: float                        # 成交量
    
    # 买卖盘信息（主要用于外汇）
    bid: Optional[float] = None          # 买价
    ask: Optional[float] = None          # 卖价
    spread: Optional[float] = None       # 买卖差价
    bid_volume: Optional[float] = None   # 买盘量
    ask_volume: Optional[float] = None   # 卖盘量
    
    # 额外信息
    tick_count: Optional[int] = None     # Tick数量（聚合数据）
    vwap: Optional[float] = None         # 成交量加权平均价
    turnover: Optional[float] = None     # 成交额
    
    # 元数据
    source: Optional[str] = None         # 数据源名称
    quality: DataQuality = DataQuality.UNKNOWN  # 数据质量
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息
    
    def __post_init__(self):
        """数据验证和计算"""
        # 计算价差
        if self.bid is not None and self.ask is not None and self.spread is None:
            self.spread = self.ask - self.bid
            
        # 验证OHLC逻辑
        if self.high < max(self.open, self.close):
            raise ValueError(f"High price {self.high} is less than max(open, close)")
            
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low price {self.low} is greater than min(open, close)")
            
        # 计算VWAP（如果没有提供）
        if self.vwap is None and self.volume > 0:
            self.vwap = (self.high + self.low + self.close) / 3
            
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'tick_count': self.tick_count,
            'vwap': self.vwap,
            'turnover': self.turnover,
            'source': self.source,
            'quality': self.quality.value,
            'metadata': self.metadata
        }


@dataclass
class DataSourceCapabilities:
    """数据源能力描述"""
    name: str                                    # 数据源名称
    supported_markets: List[MarketType]         # 支持的市场类型
    supported_intervals: List[DataInterval]     # 支持的时间间隔
    has_realtime: bool                          # 是否支持实时数据
    has_historical: bool                        # 是否支持历史数据
    has_streaming: bool = False                 # 是否支持流式数据
    requires_auth: bool = False                 # 是否需要认证
    is_free: bool = True                       # 是否免费
    
    # 数据限制
    max_history_days: Optional[int] = None      # 历史数据最大天数
    min_interval: Optional[DataInterval] = None # 最小时间间隔
    max_symbols_per_request: Optional[int] = None  # 单次请求最大标的数
    
    # 速率限制
    rate_limits: Dict[str, int] = field(default_factory=dict)  # 速率限制
    
    # 数据质量
    data_quality: DataQuality = DataQuality.MEDIUM
    latency_ms: Optional[int] = None            # 平均延迟（毫秒）
    
    # 额外信息
    api_version: Optional[str] = None           # API版本
    documentation_url: Optional[str] = None     # 文档链接
    support_contact: Optional[str] = None       # 技术支持联系方式
    
    def supports_market(self, market: MarketType) -> bool:
        """检查是否支持指定市场"""
        return market in self.supported_markets
        
    def supports_interval(self, interval: DataInterval) -> bool:
        """检查是否支持指定时间间隔"""
        return interval in self.supported_intervals
        
    def get_rate_limit(self, endpoint: str) -> Optional[int]:
        """获取指定端点的速率限制"""
        return self.rate_limits.get(endpoint)


@dataclass
class ConnectionStatus:
    """连接状态信息"""
    is_connected: bool = False
    connected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    health_score: float = 1.0  # 0-1之间，1表示完全健康


class AbstractDataSource(ABC):
    """数据源抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据源
        
        Args:
            config: 数据源配置参数
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.logger = self._setup_logger()
        self.connection_status = ConnectionStatus()
        
        # 缓存和性能相关
        self._cache = {}
        self._stats = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'last_request_time': None
        }
        
        # 速率限制
        self._rate_limiter = None
        self._last_request_times = {}
        
        self.logger.info(f"Initialized {self.name} data source")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"DataSource.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def connect(self) -> bool:
        """
        建立连接
        
        Returns:
            bool: 连接是否成功
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期  
            interval: 数据间隔
            
        Returns:
            pd.DataFrame: 历史数据，索引为时间戳
        """
        pass
    
    @abstractmethod
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """
        获取实时数据
        
        Args:
            symbols: 标的代码或代码列表
            
        Returns:
            MarketData或MarketData列表
        """
        pass
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: Callable[[Union[MarketData, List[MarketData]]], None],
        interval_seconds: float = 1.0
    ) -> None:
        """
        流式接收实时数据（默认实现使用轮询）
        
        Args:
            symbols: 标的代码列表
            callback: 数据回调函数
            interval_seconds: 轮询间隔（秒）
        """
        import threading
        import time
        
        def poll_data():
            self._streaming = True
            while self._streaming:
                try:
                    data = self.fetch_realtime_data(symbols)
                    callback(data)
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"Streaming error: {e}")
                    time.sleep(interval_seconds * 2)  # 错误时加倍等待
        
        self._streaming_thread = threading.Thread(target=poll_data, daemon=True)
        self._streaming_thread.start()
        self.logger.info(f"Started streaming for {len(symbols)} symbols")
    
    def stop_streaming(self) -> None:
        """停止流式数据接收"""
        if hasattr(self, '_streaming'):
            self._streaming = False
            self.logger.info("Stopped streaming")
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证标的代码是否有效
        
        Args:
            symbol: 标的代码
            
        Returns:
            bool: 是否有效
        """
        pass
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        搜索标的（默认实现返回空列表）
        
        Args:
            query: 搜索关键词
            limit: 最大返回数量
            
        Returns:
            List[Dict]: 匹配的标的列表
        """
        return []
    
    @abstractmethod
    def get_capabilities(self) -> DataSourceCapabilities:
        """
        获取数据源能力信息
        
        Returns:
            DataSourceCapabilities: 数据源能力描述
        """
        pass
    
    def get_supported_intervals(self) -> List[DataInterval]:
        """获取支持的时间间隔"""
        return self.get_capabilities().supported_intervals
    
    def get_supported_markets(self) -> List[MarketType]:
        """获取支持的市场类型"""  
        return self.get_capabilities().supported_markets
    
    def get_connection_status(self) -> ConnectionStatus:
        """获取连接状态"""
        return self.connection_status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取使用统计"""
        return self._stats.copy()
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 数据源是否健康
        """
        try:
            # 基本连接检查
            if not self.connection_status.is_connected:
                return False
                
            # 尝试获取一个常见标的的数据
            test_symbols = ['EURUSD', 'AAPL', 'BTC-USD']
            for symbol in test_symbols:
                if self.validate_symbol(symbol):
                    self.fetch_realtime_data(symbol)
                    return True
                    
            return False
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化标的代码（子类可重写）
        
        Args:
            symbol: 原始标的代码
            
        Returns:
            str: 标准化后的标的代码
        """
        return symbol.upper().strip()
    
    def _apply_rate_limit(self, endpoint: str) -> None:
        """
        应用速率限制
        
        Args:
            endpoint: 端点名称
        """
        capabilities = self.get_capabilities()
        limit = capabilities.get_rate_limit(endpoint)
        
        if limit is not None:
            current_time = datetime.now()
            last_time = self._last_request_times.get(endpoint)
            
            if last_time is not None:
                time_diff = (current_time - last_time).total_seconds()
                min_interval = 60.0 / limit  # 转换为每次请求的最小间隔
                
                if time_diff < min_interval:
                    import time
                    sleep_time = min_interval - time_diff
                    time.sleep(sleep_time)
            
            self._last_request_times[endpoint] = current_time
    
    def _update_stats(self, success: bool, response_time: float) -> None:
        """
        更新统计信息
        
        Args:
            success: 请求是否成功
            response_time: 响应时间（秒）
        """
        self._stats['requests_total'] += 1
        self._stats['last_request_time'] = datetime.now()
        
        if success:
            self._stats['requests_success'] += 1
        else:
            self._stats['requests_failed'] += 1
        
        # 更新平均响应时间
        total_requests = self._stats['requests_total']
        current_avg = self._stats['avg_response_time']
        self._stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # 更新健康分数
        success_rate = self._stats['requests_success'] / total_requests
        self.connection_status.health_score = success_rate
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def __repr__(self) -> str:
        """字符串表示"""
        status = "connected" if self.connection_status.is_connected else "disconnected"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"