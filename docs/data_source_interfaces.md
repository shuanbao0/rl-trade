# 数据源接口定义和实现示例

## 1. 核心接口定义

### 1.1 抽象数据源接口

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class DataInterval(Enum):
    """数据间隔枚举"""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class MarketType(Enum):
    """市场类型枚举"""
    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"

@dataclass
class DataSourceCapabilities:
    """数据源能力描述"""
    name: str
    supported_markets: List[MarketType]
    supported_intervals: List[DataInterval]
    has_realtime: bool
    has_historical: bool
    requires_auth: bool
    max_history_days: Optional[int]
    rate_limits: Optional[Dict[str, int]]  # 请求限制
    
@dataclass
class MarketData:
    """统一的市场数据格式"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    tick_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class AbstractDataSource(ABC):
    """数据源抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据源
        
        Args:
            config: 数据源配置
        """
        self.config = config
        self.logger = self._setup_logger()
        self._rate_limiter = None
        self._session = None
        
    @abstractmethod
    def connect(self) -> bool:
        """
        建立连接
        
        Returns:
            连接是否成功
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
            历史数据DataFrame
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
            实时市场数据
        """
        pass
    
    @abstractmethod
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable
    ) -> None:
        """
        流式接收实时数据
        
        Args:
            symbols: 标的代码列表
            callback: 数据回调函数
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证标的代码是否有效
        
        Args:
            symbol: 标的代码
            
        Returns:
            是否有效
        """
        pass
    
    @abstractmethod
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """
        搜索标的
        
        Args:
            query: 搜索关键词
            
        Returns:
            匹配的标的列表
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> DataSourceCapabilities:
        """
        获取数据源能力信息
        
        Returns:
            数据源能力描述
        """
        pass
    
    def get_supported_intervals(self) -> List[DataInterval]:
        """获取支持的时间间隔"""
        return self.get_capabilities().supported_intervals
    
    def get_supported_markets(self) -> List[MarketType]:
        """获取支持的市场类型"""
        return self.get_capabilities().supported_markets
```

### 1.2 数据源工厂接口

```python
from typing import Type, Dict, Optional
import importlib

class DataSourceRegistry:
    """数据源注册表"""
    
    _sources: Dict[str, Type[AbstractDataSource]] = {}
    
    @classmethod
    def register(cls, name: str, source_class: Type[AbstractDataSource]):
        """注册数据源"""
        cls._sources[name.lower()] = source_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[AbstractDataSource]]:
        """获取数据源类"""
        return cls._sources.get(name.lower())
    
    @classmethod
    def list_sources(cls) -> List[str]:
        """列出所有注册的数据源"""
        return list(cls._sources.keys())

class DataSourceFactory:
    """数据源工厂"""
    
    @staticmethod
    def create_data_source(
        source_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AbstractDataSource:
        """
        创建数据源实例
        
        Args:
            source_type: 数据源类型
            config: 配置参数
            
        Returns:
            数据源实例
        """
        source_class = DataSourceRegistry.get(source_type)
        if not source_class:
            raise ValueError(f"Unknown data source type: {source_type}")
        
        return source_class(config or {})
    
    @staticmethod
    def create_from_config(config_path: str) -> AbstractDataSource:
        """
        从配置文件创建数据源
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            数据源实例
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        source_type = config.get('type')
        return DataSourceFactory.create_data_source(source_type, config)
```

## 2. 具体实现示例

### 2.1 YFinance数据源实现

```python
import yfinance as yf
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class YFinanceDataSource(AbstractDataSource):
    """YFinance数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.proxy = config.get('proxy')
        if self.proxy:
            yf.set_config(proxy=self.proxy)
    
    def connect(self) -> bool:
        """YFinance无需显式连接"""
        return True
    
    def disconnect(self) -> None:
        """YFinance无需显式断开"""
        pass
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """获取历史数据"""
        # 转换间隔格式
        interval_map = {
            DataInterval.MINUTE_1: "1m",
            DataInterval.MINUTE_5: "5m",
            DataInterval.MINUTE_15: "15m",
            DataInterval.MINUTE_30: "30m",
            DataInterval.HOUR_1: "1h",
            DataInterval.DAY_1: "1d",
            DataInterval.WEEK_1: "1wk",
            DataInterval.MONTH_1: "1mo"
        }
        
        yf_interval = interval_map.get(interval, "1d")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=yf_interval
        )
        
        return self._standardize_dataframe(data)
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """获取实时数据"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        results = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=info.get('regularMarketOpen', 0),
                high=info.get('regularMarketDayHigh', 0),
                low=info.get('regularMarketDayLow', 0),
                close=info.get('regularMarketPrice', 0),
                volume=info.get('regularMarketVolume', 0),
                bid=info.get('bid', None),
                ask=info.get('ask', None),
                spread=info.get('ask', 0) - info.get('bid', 0) if info.get('bid') and info.get('ask') else None
            )
            results.append(market_data)
        
        return results[0] if len(results) == 1 else results
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable
    ) -> None:
        """YFinance不支持真正的流式数据，使用轮询模拟"""
        import time
        import threading
        
        def poll():
            while self._streaming:
                data = self.fetch_realtime_data(symbols)
                callback(data)
                time.sleep(self.config.get('poll_interval', 5))
        
        self._streaming = True
        thread = threading.Thread(target=poll)
        thread.daemon = True
        thread.start()
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证股票代码"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'symbol' in info
        except:
            return False
    
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """搜索股票"""
        # YFinance没有直接的搜索API，返回空列表
        return []
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="YFinance",
            supported_markets=[MarketType.STOCK, MarketType.FOREX],
            supported_intervals=[
                DataInterval.MINUTE_1, DataInterval.MINUTE_5,
                DataInterval.MINUTE_15, DataInterval.MINUTE_30,
                DataInterval.HOUR_1, DataInterval.DAY_1,
                DataInterval.WEEK_1, DataInterval.MONTH_1
            ],
            has_realtime=True,
            has_historical=True,
            requires_auth=False,
            max_history_days=None,  # 理论上无限制
            rate_limits={"requests_per_minute": 2000}
        )
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return df

# 注册YFinance数据源
DataSourceRegistry.register("yfinance", YFinanceDataSource)
```

### 2.2 TrueFX数据源实现

```python
import requests
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import csv
import io

class TrueFXDataSource(AbstractDataSource):
    """TrueFX外汇数据源实现"""
    
    BASE_URL = "https://webrates.truefx.com/rates"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.username = config.get('username')
        self.password = config.get('password')
        self.session_id = None
        self.authenticated = False
    
    def connect(self) -> bool:
        """建立认证连接"""
        if not self.username or not self.password:
            # 未认证模式，访问受限
            self.authenticated = False
            return True
        
        # 认证请求
        params = {
            'u': self.username,
            'p': self.password,
            'q': 'ozrates',
            'f': 'csv'
        }
        
        response = requests.get(f"{self.BASE_URL}/connect.html", params=params)
        
        if response.status_code == 200:
            self.session_id = response.text.strip()
            if self.session_id and self.session_id != 'error':
                self.authenticated = True
                return True
        
        return False
    
    def disconnect(self) -> None:
        """断开连接"""
        if self.session_id:
            params = {'di': self.session_id}
            requests.get(f"{self.BASE_URL}/disconnect.html", params=params)
            self.session_id = None
            self.authenticated = False
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """
        获取历史数据
        注：TrueFX主要提供实时数据，历史数据需要从其下载页面获取
        """
        # TrueFX历史数据通过文件下载提供
        # 这里实现一个占位符，实际使用时需要预先下载数据
        raise NotImplementedError(
            "TrueFX historical data requires manual download from "
            "https://www.truefx.com/truefx-historical-downloads/"
        )
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """获取实时外汇数据"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # 构建请求
        params = {
            'f': 'csv',
            'c': ','.join(symbols)
        }
        
        if self.session_id:
            params['id'] = self.session_id
        
        response = requests.get(f"{self.BASE_URL}/connect.html", params=params)
        
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch data: {response.status_code}")
        
        # 解析CSV响应
        results = []
        csv_reader = csv.reader(io.StringIO(response.text))
        
        for row in csv_reader:
            if len(row) >= 9:
                market_data = MarketData(
                    symbol=row[0],
                    timestamp=datetime.fromtimestamp(int(row[1]) / 1000),
                    open=0,  # TrueFX不提供OHLC
                    high=0,
                    low=0,
                    close=(float(row[2]) + float(row[4])) / 2,  # 使用买卖中价
                    volume=0,
                    bid=float(row[2]),
                    ask=float(row[4]),
                    spread=float(row[4]) - float(row[2]),
                    bid_volume=float(row[3]) if row[3] else None,
                    ask_volume=float(row[5]) if row[5] else None
                )
                results.append(market_data)
        
        return results[0] if len(results) == 1 else results
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable
    ) -> None:
        """流式接收实时数据"""
        import threading
        import time
        
        def stream():
            while self._streaming:
                try:
                    data = self.fetch_realtime_data(symbols)
                    callback(data)
                    time.sleep(0.5)  # TrueFX更新频率较高
                except Exception as e:
                    self.logger.error(f"Stream error: {e}")
                    time.sleep(5)
        
        self._streaming = True
        thread = threading.Thread(target=stream)
        thread.daemon = True
        thread.start()
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证外汇对"""
        valid_pairs = [
            'EUR/USD', 'USD/JPY', 'GBP/USD', 'EUR/GBP', 'USD/CHF',
            'EUR/JPY', 'EUR/CHF', 'USD/CAD', 'AUD/USD', 'GBP/JPY'
        ]
        
        if self.authenticated:
            # 认证用户可访问更多货币对
            valid_pairs.extend([
                'AUD/NZD', 'CAD/CHF', 'CHF/JPY', 'EUR/AUD', 'EUR/CAD'
            ])
        
        return symbol.upper() in valid_pairs
    
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """搜索外汇对"""
        all_pairs = [
            {'symbol': 'EUR/USD', 'name': 'Euro/US Dollar'},
            {'symbol': 'USD/JPY', 'name': 'US Dollar/Japanese Yen'},
            {'symbol': 'GBP/USD', 'name': 'British Pound/US Dollar'},
            # ... 更多
        ]
        
        query = query.upper()
        return [p for p in all_pairs if query in p['symbol']]
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="TrueFX",
            supported_markets=[MarketType.FOREX],
            supported_intervals=[DataInterval.TICK],
            has_realtime=True,
            has_historical=False,  # 需要手动下载
            requires_auth=False,  # 可选认证
            max_history_days=None,
            rate_limits=None  # 无明确限制
        )

# 注册TrueFX数据源
DataSourceRegistry.register("truefx", TrueFXDataSource)
```

### 2.3 Oanda数据源实现

```python
from oandapyV20 import API
from oandapyV20.endpoints import instruments, pricing
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class OandaDataSource(AbstractDataSource):
    """Oanda外汇数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.account_id = config.get('account_id')
        self.access_token = config.get('access_token')
        self.environment = config.get('environment', 'practice')
        self.api = None
    
    def connect(self) -> bool:
        """建立API连接"""
        if not self.access_token:
            raise ValueError("Oanda requires access_token")
        
        self.api = API(
            access_token=self.access_token,
            environment=self.environment
        )
        return True
    
    def disconnect(self) -> None:
        """断开连接"""
        self.api = None
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """获取历史数据"""
        # 转换间隔格式
        granularity_map = {
            DataInterval.SECOND_1: "S1",
            DataInterval.MINUTE_1: "M1",
            DataInterval.MINUTE_5: "M5",
            DataInterval.MINUTE_15: "M15",
            DataInterval.MINUTE_30: "M30",
            DataInterval.HOUR_1: "H1",
            DataInterval.HOUR_4: "H4",
            DataInterval.DAY_1: "D",
            DataInterval.WEEK_1: "W",
            DataInterval.MONTH_1: "M"
        }
        
        params = {
            "from": start_date.isoformat() if isinstance(start_date, datetime) else start_date,
            "to": end_date.isoformat() if isinstance(end_date, datetime) else end_date,
            "granularity": granularity_map.get(interval, "H1"),
            "price": "MBA"  # Mid, Bid, Ask
        }
        
        request = instruments.InstrumentsCandles(
            instrument=symbol,
            params=params
        )
        
        response = self.api.request(request)
        
        # 转换为DataFrame
        data = []
        for candle in response['candles']:
            if candle['complete']:
                data.append({
                    'timestamp': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume']),
                    'bid_open': float(candle['bid']['o']),
                    'bid_close': float(candle['bid']['c']),
                    'ask_open': float(candle['ask']['o']),
                    'ask_close': float(candle['ask']['c'])
                })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """获取实时数据"""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        params = {
            "instruments": ",".join(symbols)
        }
        
        request = pricing.PricingInfo(
            accountID=self.account_id,
            params=params
        )
        
        response = self.api.request(request)
        
        results = []
        for price in response['prices']:
            market_data = MarketData(
                symbol=price['instrument'],
                timestamp=datetime.fromisoformat(price['time'].replace('Z', '+00:00')),
                open=0,  # 实时数据无OHLC
                high=0,
                low=0,
                close=float(price['closeoutBid']) if 'closeoutBid' in price else 0,
                volume=0,
                bid=float(price['bids'][0]['price']) if price['bids'] else None,
                ask=float(price['asks'][0]['price']) if price['asks'] else None,
                spread=float(price['asks'][0]['price']) - float(price['bids'][0]['price']) 
                       if price['bids'] and price['asks'] else None
            )
            results.append(market_data)
        
        return results[0] if len(results) == 1 else results
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable
    ) -> None:
        """流式接收实时数据"""
        from oandapyV20.endpoints import pricing
        
        params = {
            "instruments": ",".join(symbols)
        }
        
        request = pricing.PricingStream(
            accountID=self.account_id,
            params=params
        )
        
        response = self.api.request(request)
        
        for msg in response:
            if msg['type'] == 'PRICE':
                market_data = MarketData(
                    symbol=msg['instrument'],
                    timestamp=datetime.fromisoformat(msg['time'].replace('Z', '+00:00')),
                    open=0,
                    high=0,
                    low=0,
                    close=float(msg['closeoutBid']) if 'closeoutBid' in msg else 0,
                    volume=0,
                    bid=float(msg['bids'][0]['price']) if msg['bids'] else None,
                    ask=float(msg['asks'][0]['price']) if msg['asks'] else None
                )
                callback(market_data)
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证交易对"""
        try:
            request = instruments.InstrumentsDetails(instrument=symbol)
            self.api.request(request)
            return True
        except:
            return False
    
    def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """搜索交易对"""
        request = instruments.InstrumentsList(accountID=self.account_id)
        response = self.api.request(request)
        
        results = []
        query = query.upper()
        
        for instrument in response['instruments']:
            if query in instrument['name'] or query in instrument['displayName']:
                results.append({
                    'symbol': instrument['name'],
                    'name': instrument['displayName'],
                    'type': instrument['type']
                })
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="Oanda",
            supported_markets=[MarketType.FOREX],
            supported_intervals=[
                DataInterval.SECOND_1, DataInterval.MINUTE_1,
                DataInterval.MINUTE_5, DataInterval.MINUTE_15,
                DataInterval.MINUTE_30, DataInterval.HOUR_1,
                DataInterval.HOUR_4, DataInterval.DAY_1,
                DataInterval.WEEK_1, DataInterval.MONTH_1
            ],
            has_realtime=True,
            has_historical=True,
            requires_auth=True,
            max_history_days=1825,  # 约5年
            rate_limits={"requests_per_second": 120}
        )

# 注册Oanda数据源
DataSourceRegistry.register("oanda", OandaDataSource)
```

## 3. 数据源聚合器

```python
class MultiSourceAggregator:
    """多数据源聚合器"""
    
    def __init__(self, sources: List[str], config: Dict[str, Any] = None):
        """
        初始化聚合器
        
        Args:
            sources: 数据源列表
            config: 配置
        """
        self.sources = []
        for source_name in sources:
            source = DataSourceFactory.create_data_source(
                source_name,
                config.get(source_name, {}) if config else {}
            )
            self.sources.append(source)
    
    def get_best_price(self, symbol: str) -> Dict[str, Any]:
        """
        获取最优价格
        
        Args:
            symbol: 标的代码
            
        Returns:
            最优报价信息
        """
        best_bid = None
        best_ask = None
        best_bid_source = None
        best_ask_source = None
        
        for source in self.sources:
            try:
                data = source.fetch_realtime_data(symbol)
                if data.bid and (not best_bid or data.bid > best_bid):
                    best_bid = data.bid
                    best_bid_source = source.__class__.__name__
                
                if data.ask and (not best_ask or data.ask < best_ask):
                    best_ask = data.ask
                    best_ask_source = source.__class__.__name__
            except:
                continue
        
        return {
            'symbol': symbol,
            'best_bid': best_bid,
            'best_bid_source': best_bid_source,
            'best_ask': best_ask,
            'best_ask_source': best_ask_source,
            'spread': best_ask - best_bid if best_bid and best_ask else None
        }
    
    def aggregate_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: DataInterval
    ) -> pd.DataFrame:
        """
        聚合多源历史数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            聚合后的数据
        """
        all_data = []
        
        for source in self.sources:
            try:
                data = source.fetch_historical_data(
                    symbol, start_date, end_date, interval
                )
                data['source'] = source.__class__.__name__
                all_data.append(data)
            except:
                continue
        
        if not all_data:
            raise ValueError("No data available from any source")
        
        # 合并数据
        combined = pd.concat(all_data)
        
        # 按时间戳分组，取平均值
        aggregated = combined.groupby(combined.index).agg({
            'open': 'mean',
            'high': 'max',
            'low': 'min',
            'close': 'mean',
            'volume': 'sum'
        })
        
        return aggregated
```

## 4. 使用示例

```python
# 基本使用
from src.data.sources import DataSourceFactory, DataInterval

# 创建单个数据源
yfinance_source = DataSourceFactory.create_data_source(
    'yfinance',
    {'proxy': 'socks5://127.0.0.1:7891'}
)

# 获取历史数据
data = yfinance_source.fetch_historical_data(
    'AAPL',
    '2024-01-01',
    '2024-12-31',
    DataInterval.DAY_1
)

# 创建Oanda数据源
oanda_source = DataSourceFactory.create_data_source(
    'oanda',
    {
        'account_id': 'your_account',
        'access_token': 'your_token',
        'environment': 'practice'
    }
)

# 获取实时外汇数据
forex_data = oanda_source.fetch_realtime_data(['EUR_USD', 'GBP_USD'])

# 使用聚合器
aggregator = MultiSourceAggregator(
    ['truefx', 'oanda'],
    {
        'truefx': {'username': 'user', 'password': 'pass'},
        'oanda': {'account_id': 'account', 'access_token': 'token'}
    }
)

# 获取最优价格
best_price = aggregator.get_best_price('EUR_USD')
print(f"Best bid: {best_price['best_bid']} from {best_price['best_bid_source']}")
print(f"Best ask: {best_price['best_ask']} from {best_price['best_ask_source']}")

# 流式数据接收
def on_price_update(data):
    print(f"New price: {data.symbol} - Bid: {data.bid}, Ask: {data.ask}")

oanda_source.stream_realtime_data(['EUR_USD'], on_price_update)
```