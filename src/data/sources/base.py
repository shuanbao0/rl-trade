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
    TICK = "tick"          # Tick级数据（最高频）
    SECOND_1 = "1s"        # 1秒K线
    SECOND_5 = "5s"        # 5秒K线
    SECOND_15 = "15s"      # 15秒K线
    SECOND_30 = "30s"      # 30秒K线
    MINUTE_1 = "1m"        # 1分钟K线
    MINUTE_2 = "2m"        # 2分钟K线
    MINUTE_5 = "5m"        # 5分钟K线
    MINUTE_15 = "15m"      # 15分钟K线
    MINUTE_30 = "30m"      # 30分钟K线
    MINUTE_90 = "90m"      # 90分钟K线
    HOUR_1 = "1h"          # 1小时K线
    HOUR_4 = "4h"          # 4小时K线
    HOUR_6 = "6h"          # 6小时K线
    HOUR_8 = "8h"          # 8小时K线
    HOUR_12 = "12h"        # 12小时K线
    DAY_1 = "1d"           # 日K线
    DAY_3 = "3d"           # 3日K线
    WEEK_1 = "1w"          # 周K线
    MONTH_1 = "1M"         # 月K线
    MONTH_3 = "3M"         # 季度K线
    YEAR_1 = "1Y"          # 年K线


class MarketType(Enum):
    """市场类型枚举"""
    STOCK = "stock"               # 股票市场
    FOREX = "forex"               # 外汇市场
    CRYPTO = "crypto"             # 加密货币市场
    FUTURES = "futures"           # 期货市场
    OPTIONS = "options"           # 期权市场
    COMMODITIES = "commodities"   # 商品市场
    BONDS = "bonds"               # 债券市场
    INDEX = "index"               # 指数
    ETF = "etf"                   # 交易所交易基金


class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"        # 高质量：官方API，实时更新，数据准确
    MEDIUM = "medium"    # 中等质量：延迟较小，数据完整性良好
    LOW = "low"         # 低质量：延迟较大，可能有数据缺失
    UNKNOWN = "unknown"  # 未知质量：无法确定数据质量等级


class DataSource(Enum):
    """数据源枚举"""
    YFINANCE = "yfinance"     # Yahoo Finance - 免费股票/加密货币数据
    TRUEFX = "truefx"         # TrueFX - 免费外汇数据，需要注册
    OANDA = "oanda"           # Oanda - 专业外汇/CFD数据，需要API密钥
    FXMINUTE = "fxminute"     # FX-1-Minute-Data - 本地外汇历史数据
    HISTDATA = "histdata"     # HistData - 外汇历史数据文件
    AUTO = "auto"             # 自动选择最优数据源
    
    @classmethod
    def from_string(cls, value: str) -> 'DataSource':
        """从字符串创建数据源枚举，支持向后兼容"""
        if isinstance(value, cls):
            return value
        
        # 标准化字符串
        value = str(value).lower().strip()
        
        # 尝试直接匹配
        for source in cls:
            if source.value == value:
                return source
        
        # 容错匹配
        mapping = {
            'yahoo': cls.YFINANCE,
            'yf': cls.YFINANCE,
            'yahoo_finance': cls.YFINANCE,
            'truefx': cls.TRUEFX,
            'true_fx': cls.TRUEFX,
            'oanda': cls.OANDA,
            'fxminute': cls.FXMINUTE,
            'fx_minute': cls.FXMINUTE,
            'fx-minute': cls.FXMINUTE,
            'fxminutedata': cls.FXMINUTE,
            'histdata': cls.HISTDATA,
            'hist_data': cls.HISTDATA,
            'historical_data': cls.HISTDATA,
            'automatic': cls.AUTO,
            'default': cls.AUTO,
        }
        
        if value in mapping:
            return mapping[value]
        
        raise ValueError(f"Unknown data source: {value}. Available sources: {[s.value for s in cls]}")
    
    @property
    def display_name(self) -> str:
        """获取友好的显示名称"""
        display_names = {
            self.YFINANCE: "Yahoo Finance",
            self.TRUEFX: "TrueFX",
            self.OANDA: "Oanda",
            self.FXMINUTE: "FX-1-Minute-Data",
            self.HISTDATA: "HistData",
            self.AUTO: "Auto-Select"
        }
        return display_names.get(self, self.value.title())
    
    @property
    def supported_markets(self) -> List[MarketType]:
        """获取支持的市场类型"""
        support_mapping = {
            self.YFINANCE: [MarketType.STOCK, MarketType.CRYPTO, MarketType.ETF, MarketType.INDEX],
            self.TRUEFX: [MarketType.FOREX],
            self.OANDA: [MarketType.FOREX, MarketType.COMMODITIES, MarketType.INDEX],
            self.FXMINUTE: [MarketType.FOREX],
            self.HISTDATA: [MarketType.FOREX],
            self.AUTO: list(MarketType)  # AUTO支持所有市场类型
        }
        return support_mapping.get(self, [])
    
    @property
    def data_quality(self) -> DataQuality:
        """获取数据源的典型质量等级"""
        quality_mapping = {
            self.YFINANCE: DataQuality.MEDIUM,
            self.TRUEFX: DataQuality.HIGH,
            self.OANDA: DataQuality.HIGH,
            self.FXMINUTE: DataQuality.HIGH,
            self.HISTDATA: DataQuality.MEDIUM,
            self.AUTO: DataQuality.UNKNOWN
        }
        return quality_mapping.get(self, DataQuality.UNKNOWN)


class DataPeriod(Enum):
    """数据周期枚举 - 定义标准的时间周期"""
    
    # 天数周期
    DAYS_1 = "1d"       # 1天
    DAYS_7 = "7d"       # 7天（1周）
    DAYS_30 = "30d"     # 30天（约1个月）
    DAYS_60 = "60d"     # 60天（约2个月）
    DAYS_90 = "90d"     # 90天（约3个月）
    
    # 周数周期
    WEEK_1 = "1w"       # 1周
    WEEK_2 = "2w"       # 2周
    WEEK_4 = "4w"       # 4周
    
    # 月数周期
    MONTH_1 = "1mo"     # 1个月
    MONTH_3 = "3mo"     # 3个月
    MONTH_6 = "6mo"     # 6个月
    MONTH_12 = "12mo"   # 12个月
    
    # 年数周期
    YEAR_1 = "1y"       # 1年
    YEAR_2 = "2y"       # 2年
    YEAR_5 = "5y"       # 5年
    YEAR_10 = "10y"     # 10年
    
    # 特殊周期
    MAX = "max"         # 最大可用历史数据
    
    @classmethod
    def from_string(cls, value: str) -> 'DataPeriod':
        """从字符串创建数据周期枚举，支持向后兼容"""
        if isinstance(value, cls):
            return value
        
        # 标准化字符串
        value = str(value).lower().strip()
        
        # 尝试直接匹配
        for period in cls:
            if period.value == value:
                return period
        
        # 容错匹配和别名支持
        mapping = {
            # 天数别名
            '1': cls.DAYS_1,
            '1day': cls.DAYS_1,
            '1days': cls.DAYS_1,
            '7': cls.DAYS_7,
            '7day': cls.DAYS_7,
            '7days': cls.DAYS_7,
            '30': cls.DAYS_30,
            '30day': cls.DAYS_30,
            '30days': cls.DAYS_30,
            '60': cls.DAYS_60,
            '60day': cls.DAYS_60,
            '60days': cls.DAYS_60,
            '90': cls.DAYS_90,
            '90day': cls.DAYS_90,
            '90days': cls.DAYS_90,
            
            # 周数别名
            '1week': cls.WEEK_1,
            '1weeks': cls.WEEK_1,
            '2week': cls.WEEK_2,
            '2weeks': cls.WEEK_2,
            '4week': cls.WEEK_4,
            '4weeks': cls.WEEK_4,
            
            # 月数别名
            '1m': cls.MONTH_1,    # 注意：这里可能与分钟间隔冲突，需要上下文判断
            '1month': cls.MONTH_1,
            '1months': cls.MONTH_1,
            '3m': cls.MONTH_3,
            '3month': cls.MONTH_3,
            '3months': cls.MONTH_3,
            '6m': cls.MONTH_6,
            '6month': cls.MONTH_6,
            '6months': cls.MONTH_6,
            '12m': cls.MONTH_12,
            '12month': cls.MONTH_12,
            '12months': cls.MONTH_12,
            
            # 年数别名
            '1year': cls.YEAR_1,
            '1years': cls.YEAR_1,
            '2year': cls.YEAR_2,
            '2years': cls.YEAR_2,
            '5year': cls.YEAR_5,
            '5years': cls.YEAR_5,
            '10year': cls.YEAR_10,
            '10years': cls.YEAR_10,
            
            # 特殊值别名
            'maximum': cls.MAX,
            'all': cls.MAX,
            'full': cls.MAX,
            'complete': cls.MAX,
        }
        
        if value in mapping:
            return mapping[value]
        
        # 尝试解析数字+单位格式（如"365d", "24mo", "3y"等）
        import re
        pattern = r'^(\d+)(d|day|days|w|week|weeks|mo|month|months|y|year|years)$'
        match = re.match(pattern, value)
        
        if match:
            number, unit = match.groups()
            number = int(number)
            
            # 根据单位创建对应的周期
            if unit in ['d', 'day', 'days']:
                # 对于常见天数，返回预定义的枚举值
                if number == 1: return cls.DAYS_1
                elif number == 7: return cls.DAYS_7
                elif number == 30: return cls.DAYS_30
                elif number == 60: return cls.DAYS_60
                elif number == 90: return cls.DAYS_90
                # 对于其他天数，返回最接近的预定义值
                else:
                    if number <= 3: return cls.DAYS_1
                    elif number <= 14: return cls.DAYS_7
                    elif number <= 45: return cls.DAYS_30
                    elif number <= 75: return cls.DAYS_60
                    else: return cls.DAYS_90
                    
            elif unit in ['w', 'week', 'weeks']:
                if number == 1: return cls.WEEK_1
                elif number == 2: return cls.WEEK_2
                elif number >= 4: return cls.WEEK_4
                else: return cls.WEEK_1
                
            elif unit in ['mo', 'month', 'months']:
                if number == 1: return cls.MONTH_1
                elif number <= 3: return cls.MONTH_3
                elif number <= 6: return cls.MONTH_6
                else: return cls.MONTH_12
                
            elif unit in ['y', 'year', 'years']:
                if number == 1: return cls.YEAR_1
                elif number == 2: return cls.YEAR_2
                elif number <= 5: return cls.YEAR_5
                else: return cls.YEAR_10
        
        raise ValueError(f"Unknown data period: {value}. Available periods: {[p.value for p in cls]}")
    
    @property
    def display_name(self) -> str:
        """获取友好的显示名称"""
        display_names = {
            self.DAYS_1: "1 Day",
            self.DAYS_7: "7 Days (1 Week)", 
            self.DAYS_30: "30 Days (~1 Month)",
            self.DAYS_60: "60 Days (~2 Months)",
            self.DAYS_90: "90 Days (~3 Months)",
            
            self.WEEK_1: "1 Week",
            self.WEEK_2: "2 Weeks", 
            self.WEEK_4: "4 Weeks (~1 Month)",
            
            self.MONTH_1: "1 Month",
            self.MONTH_3: "3 Months",
            self.MONTH_6: "6 Months",
            self.MONTH_12: "12 Months (1 Year)",
            
            self.YEAR_1: "1 Year",
            self.YEAR_2: "2 Years",
            self.YEAR_5: "5 Years", 
            self.YEAR_10: "10 Years",
            
            self.MAX: "Maximum Available Data"
        }
        return display_names.get(self, self.value.title())
    
    def to_days(self) -> int:
        """转换为天数"""
        if self == self.MAX:
            return 365 * 20  # 假设最大20年
        
        day_mapping = {
            self.DAYS_1: 1,
            self.DAYS_7: 7,
            self.DAYS_30: 30,
            self.DAYS_60: 60,
            self.DAYS_90: 90,
            
            self.WEEK_1: 7,
            self.WEEK_2: 14,
            self.WEEK_4: 28,
            
            self.MONTH_1: 30,
            self.MONTH_3: 90,
            self.MONTH_6: 180,
            self.MONTH_12: 365,
            
            self.YEAR_1: 365,
            self.YEAR_2: 730,
            self.YEAR_5: 1825,
            self.YEAR_10: 3650,
        }
        return day_mapping.get(self, 365)
    
    @property 
    def is_short_term(self) -> bool:
        """是否为短期周期（<=30天）"""
        return self.to_days() <= 30
    
    @property
    def is_medium_term(self) -> bool:
        """是否为中期周期（30天-1年）"""
        days = self.to_days()
        return 30 < days <= 365
    
    @property
    def is_long_term(self) -> bool:
        """是否为长期周期（>1年）"""
        return self.to_days() > 365
    
    def get_recommended_interval(self, data_source: 'DataSource') -> str:
        """根据周期和数据源推荐合适的时间间隔"""
        days = self.to_days()
        
        if data_source == DataSource.YFINANCE:
            if days <= 7:
                return "1m"      # 7天内用1分钟
            elif days <= 60:
                return "5m"      # 60天内用5分钟
            elif days <= 730:
                return "1h"      # 2年内用1小时
            else:
                return "1d"      # 超过2年用日线
                
        elif data_source == DataSource.FXMINUTE:
            return "1m"          # FXMinute只支持1分钟
            
        elif data_source in [DataSource.TRUEFX, DataSource.OANDA]:
            if days <= 7:
                return "1m"
            elif days <= 30: 
                return "5m"
            elif days <= 180:
                return "1h"
            else:
                return "1d"
                
        else:
            # 默认推荐
            if days <= 7:
                return "1m"
            elif days <= 30:
                return "5m" 
            elif days <= 365:
                return "1h"
            else:
                return "1d"


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
    
    # 市场特定字段
    market_type: Optional[MarketType] = None     # 市场类型
    interval: Optional[DataInterval] = None      # 数据间隔
    
    # 股票特定字段
    adj_close: Optional[float] = None    # 复权收盘价
    dividend: Optional[float] = None     # 分红
    split_ratio: Optional[float] = None  # 拆股比例
    
    # 外汇特定字段
    pip_value: Optional[float] = None    # 点值
    pip_spread: Optional[float] = None   # 点差
    
    # 加密货币特定字段
    quote_volume: Optional[float] = None # 计价货币成交量
    
    # 期货/期权特定字段
    open_interest: Optional[int] = None  # 持仓量
    settlement_price: Optional[float] = None  # 结算价
    expiry_date: Optional[datetime] = None    # 到期日
    
    # 元数据
    source: Optional[DataSource] = None  # 数据源名称
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
            # 市场特定字段
            'market_type': self.market_type.value if self.market_type else None,
            'interval': self.interval.value if self.interval else None,
            # 股票特定字段
            'adj_close': self.adj_close,
            'dividend': self.dividend,
            'split_ratio': self.split_ratio,
            # 外汇特定字段
            'pip_value': self.pip_value,
            'pip_spread': self.pip_spread,
            # 加密货币特定字段
            'quote_volume': self.quote_volume,
            # 期货/期权特定字段
            'open_interest': self.open_interest,
            'settlement_price': self.settlement_price,
            'expiry_date': self.expiry_date,
            # 元数据
            'source': self.source.value if self.source else None,
            'quality': self.quality.value,
            'metadata': self.metadata
        }


@dataclass
class DataSourceCapabilities:
    """数据源能力描述"""
    name: str                                    # 数据源名称
    source_id: DataSource                       # 数据源标识符
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
        # 使用统一日志系统
        from ...utils.logger import get_logger
        logger = get_logger(f"DataSource.{self.name}")
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