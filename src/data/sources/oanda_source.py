"""
Oanda数据源实现

提供外汇和差价合约数据，支持：
- 70+外汇货币对和其他金融产品
- 实时和历史价格数据
- 专业级API，需要账户认证
- 高精度价格数据和低延迟
"""

import time
import requests
import json
import threading
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base import (
    AbstractDataSource, DataInterval, MarketType, MarketData, 
    DataSourceCapabilities, DataQuality
)
from .converter import DataConverter


class OandaDataSource(AbstractDataSource):
    """Oanda外汇数据源实现"""
    
    # API环境
    PRACTICE_API_URL = "https://api-fxpractice.oanda.com"
    LIVE_API_URL = "https://api-fxtrade.oanda.com"
    
    PRACTICE_STREAM_URL = "https://stream-fxpractice.oanda.com"
    LIVE_STREAM_URL = "https://stream-fxtrade.oanda.com"
    
    # 支持的产品类型
    SUPPORTED_INSTRUMENT_TYPES = [
        'CURRENCY',  # 外汇
        'CFD',       # 差价合约
        'METAL'      # 贵金属
    ]
    
    # 数据间隔映射
    INTERVAL_MAP = {
        DataInterval.SECOND_5: 'S5',
        DataInterval.SECOND_15: 'S15',
        DataInterval.SECOND_30: 'S30',
        DataInterval.MINUTE_1: 'M1',
        DataInterval.MINUTE_2: 'M2',
        DataInterval.MINUTE_5: 'M5',
        DataInterval.MINUTE_15: 'M15',
        DataInterval.MINUTE_30: 'M30',
        DataInterval.HOUR_1: 'H1',
        DataInterval.HOUR_4: 'H4',
        DataInterval.HOUR_6: 'H6',
        DataInterval.HOUR_8: 'H8',
        DataInterval.HOUR_12: 'H12',
        DataInterval.DAY_1: 'D',
        DataInterval.WEEK_1: 'W',
        DataInterval.MONTH_1: 'M'
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 认证配置
        self.access_token = config.get('access_token')
        self.account_id = config.get('account_id')
        
        if not self.access_token:
            raise ValueError("Oanda access_token is required")
        if not self.account_id:
            raise ValueError("Oanda account_id is required")
        
        # 环境配置
        self.environment = config.get('environment', 'practice').lower()
        if self.environment == 'practice':
            self.api_url = self.PRACTICE_API_URL
            self.stream_url = self.PRACTICE_STREAM_URL
        elif self.environment == 'live':
            self.api_url = self.LIVE_API_URL
            self.stream_url = self.LIVE_STREAM_URL
        else:
            raise ValueError("environment must be 'practice' or 'live'")
        
        # 请求配置
        self.timeout = config.get('timeout', 15)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.rate_limit = config.get('rate_limit', 120)  # 每秒请求数限制
        
        # 流式数据配置
        self.stream_interval = config.get('stream_interval', 1.0)
        self._streaming = False
        self._streaming_thread = None
        
        # 请求头
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # 缓存的产品信息
        self._instruments = None
        self._instruments_cache_time = None
        self._cache_ttl = 3600  # 1小时缓存
        
        self.logger.info("Oanda data source initialized")
    
    def connect(self) -> bool:
        """建立连接"""
        try:
            # 验证账户访问
            response = requests.get(
                f"{self.api_url}/v3/accounts/{self.account_id}",
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                account_info = response.json()
                self.connection_status.is_connected = True
                self.connection_status.connected_at = datetime.now()
                self.connection_status.last_error = None
                
                currency = account_info.get('account', {}).get('currency', 'USD')
                balance = account_info.get('account', {}).get('balance', '0')
                
                self.logger.info(f"Oanda connected successfully - Account: {self.account_id}")
                self.logger.info(f"Environment: {self.environment}, Currency: {currency}, Balance: {balance}")
                return True
            else:
                raise ConnectionError(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.connection_status.is_connected = False
            self.connection_status.last_error = str(e)
            self.connection_status.retry_count += 1
            self.logger.error(f"Oanda connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.stop_streaming()
        self.connection_status.is_connected = False
        self.logger.info("Oanda disconnected")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """获取历史数据"""
        start_time = time.time()
        
        # 标准化参数
        normalized_symbol = self._normalize_symbol(symbol)
        
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}")
        
        granularity = self.INTERVAL_MAP[interval]
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        try:
            # 构建请求参数
            params = {
                'granularity': granularity,
                'from': start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'to': end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'includeFirst': 'true'
            }
            
            # 获取蜡烛图数据
            candles_data = self._fetch_candles_data(normalized_symbol, params)
            
            if not candles_data:
                raise ValueError(f"No historical data available for {symbol}")
            
            # 转换为DataFrame
            df = self._parse_candles_to_dataframe(candles_data, normalized_symbol)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.debug(f"Fetched Oanda historical data for {symbol}: {len(df)} records in {response_time:.3f}s")
            
            return df
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch Oanda historical data: {e}")
            raise
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """获取实时数据"""
        start_time = time.time()
        single_symbol = isinstance(symbols, str)
        
        if single_symbol:
            symbols = [symbols]
        
        # 标准化货币对格式
        normalized_symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        
        try:
            # 获取实时价格数据
            pricing_data = self._fetch_pricing_data(normalized_symbols)
            
            if not pricing_data:
                raise ValueError(f"No realtime data available for symbols: {symbols}")
            
            results = []
            for price_info in pricing_data:
                market_data = self._parse_pricing_data(price_info)
                if market_data:
                    results.append(market_data)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.debug(f"Fetched Oanda realtime data for {len(results)} symbols in {response_time:.3f}s")
            
            return results[0] if single_symbol and results else results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch Oanda realtime data: {e}")
            raise
    
    def _fetch_candles_data(self, symbol: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Oanda API获取蜡烛图数据"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.api_url}/v3/instruments/{symbol}/candles",
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('candles', [])
                elif response.status_code == 429:
                    # 速率限制
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Oanda rate limit hit, retrying in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        raise requests.RequestException(f"HTTP {response.status_code}: Rate limit exceeded")
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Oanda candles request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        return []
    
    def _fetch_pricing_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """从Oanda API获取实时价格数据"""
        instruments = ','.join(symbols)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.api_url}/v3/accounts/{self.account_id}/pricing",
                    headers=self.headers,
                    params={'instruments': instruments},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('prices', [])
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Oanda rate limit hit, retrying in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        raise requests.RequestException(f"HTTP {response.status_code}: Rate limit exceeded")
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Oanda pricing request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        return []
    
    def _parse_candles_to_dataframe(self, candles: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
        """将蜡烛图数据转换为DataFrame"""
        records = []
        
        for candle in candles:
            if not candle.get('complete', False):
                continue  # 跳过未完成的蜡烛图
            
            time_str = candle['time']
            timestamp = pd.to_datetime(time_str)
            
            # 获取bid/ask数据，优先使用mid数据
            if 'mid' in candle:
                ohlc_data = candle['mid']
            elif 'bid' in candle:
                ohlc_data = candle['bid']
            elif 'ask' in candle:
                ohlc_data = candle['ask']
            else:
                continue
            
            record = {
                'timestamp': timestamp,
                'open': float(ohlc_data['o']),
                'high': float(ohlc_data['h']),
                'low': float(ohlc_data['l']),
                'close': float(ohlc_data['c']),
                'volume': int(candle.get('volume', 0)),
                'symbol': symbol,
                'source': 'oanda'
            }
            
            # 添加bid/ask数据（如果有）
            if 'bid' in candle and 'ask' in candle:
                record['bid'] = float(candle['bid']['c'])
                record['ask'] = float(candle['ask']['c'])
                record['spread'] = record['ask'] - record['bid']
            
            records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _parse_pricing_data(self, price_info: Dict[str, Any]) -> Optional[MarketData]:
        """将Oanda价格数据转换为MarketData"""
        try:
            symbol = price_info['instrument']
            time_str = price_info['time']
            timestamp = pd.to_datetime(time_str)
            
            # 获取买卖价
            bids = price_info.get('bids', [])
            asks = price_info.get('asks', [])
            
            if not bids or not asks:
                return None
            
            bid = float(bids[0]['price'])
            ask = float(asks[0]['price'])
            mid_price = (bid + ask) / 2
            spread = ask - bid
            
            # 获取流动性
            bid_liquidity = int(bids[0].get('liquidity', 0))
            ask_liquidity = int(asks[0].get('liquidity', 0))
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp.to_pydatetime(),
                open=mid_price,  # Oanda实时数据没有OHLC，使用中间价
                high=mid_price,
                low=mid_price,
                close=mid_price,
                volume=0,  # 外汇现货没有传统成交量概念
                bid=bid,
                ask=ask,
                spread=spread,
                bid_volume=bid_liquidity,
                ask_volume=ask_liquidity,
                source='oanda',
                quality=DataQuality.HIGH,
                metadata={
                    'account_id': self.account_id,
                    'environment': self.environment,
                    'tradeable': price_info.get('tradeable', False),
                    'unitsAvailable': price_info.get('unitsAvailable', {})
                }
            )
            
            return market_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Oanda pricing data {price_info}: {e}")
            return None
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable,
        interval_seconds: float = None
    ) -> None:
        """流式接收实时数据"""
        if self._streaming:
            self.logger.warning("Oanda streaming already active")
            return
        
        interval_seconds = interval_seconds or self.stream_interval
        normalized_symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        instruments = ','.join(normalized_symbols)
        
        def stream_worker():
            """流式数据工作线程"""
            self._streaming = True
            self.logger.info(f"Started Oanda streaming for {len(symbols)} instruments, interval: {interval_seconds}s")
            
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self._streaming:
                try:
                    # 使用Oanda流式API
                    with requests.get(
                        f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream",
                        headers=self.headers,
                        params={'instruments': instruments},
                        stream=True,
                        timeout=self.timeout
                    ) as response:
                        
                        if response.status_code != 200:
                            raise requests.RequestException(f"HTTP {response.status_code}")
                        
                        for line in response.iter_lines():
                            if not self._streaming:
                                break
                            
                            if line:
                                try:
                                    data = json.loads(line.decode('utf-8'))
                                    
                                    if data.get('type') == 'PRICE':
                                        market_data = self._parse_pricing_data(data)
                                        if market_data:
                                            callback(market_data)
                                            consecutive_errors = 0
                                    
                                except json.JSONDecodeError:
                                    continue
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.warning(f"Oanda streaming error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive Oanda streaming errors, stopping")
                        break
                    
                    # 错误时使用指数退避
                    error_delay = min(interval_seconds * (2 ** consecutive_errors), 30)
                    time.sleep(error_delay)
            
            self._streaming = False
            self.logger.info("Oanda streaming stopped")
        
        # 启动流式数据线程
        self._streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self._streaming_thread.start()
    
    def stop_streaming(self) -> None:
        """停止流式数据接收"""
        if self._streaming:
            self._streaming = False
            if self._streaming_thread and self._streaming_thread.is_alive():
                self._streaming_thread.join(timeout=5)
            self.logger.info("Oanda streaming stopped")
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证产品代码是否支持"""
        instruments = self._get_instruments()
        normalized_symbol = self._normalize_symbol(symbol)
        return normalized_symbol in instruments
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """搜索产品代码"""
        instruments = self._get_instruments()
        
        query_upper = query.upper()
        results = []
        
        for instrument_name, info in instruments.items():
            display_name = info.get('displayName', '').upper()
            if query_upper in instrument_name or query_upper in display_name:
                results.append({
                    'symbol': instrument_name,
                    'name': info.get('displayName', instrument_name),
                    'type': info.get('type', 'unknown').lower(),
                    'source': 'oanda'
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="OANDA",
            supported_markets=[MarketType.FOREX, MarketType.COMMODITIES],
            supported_intervals=list(self.INTERVAL_MAP.keys()),
            has_realtime=True,
            has_historical=True,
            has_streaming=True,
            requires_auth=True,
            is_free=False,
            max_history_days=5000,  # Oanda提供历史数据回到2005年
            min_interval=DataInterval.SECOND_5,
            max_symbols_per_request=20,
            rate_limits={'requests_per_second': self.rate_limit},
            data_quality=DataQuality.HIGH,
            latency_ms=50,
            api_version="OANDA v20 REST API",
            documentation_url="https://developer.oanda.com/rest-live-v20/",
            support_contact="api@oanda.com"
        )
    
    def _get_instruments(self) -> Dict[str, Dict[str, Any]]:
        """获取支持的产品列表（带缓存）"""
        now = time.time()
        
        # 检查缓存
        if (self._instruments is not None and 
            self._instruments_cache_time is not None and
            now - self._instruments_cache_time < self._cache_ttl):
            return self._instruments
        
        try:
            response = requests.get(
                f"{self.api_url}/v3/accounts/{self.account_id}/instruments",
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                instruments = {}
                
                for instrument in data.get('instruments', []):
                    name = instrument['name']
                    instruments[name] = {
                        'displayName': instrument.get('displayName', name),
                        'type': instrument.get('type', 'UNKNOWN'),
                        'marginRate': instrument.get('marginRate', '0.05'),
                        'pipLocation': instrument.get('pipLocation', -4),
                        'displayPrecision': instrument.get('displayPrecision', 5)
                    }
                
                self._instruments = instruments
                self._instruments_cache_time = now
                self.logger.info(f"Loaded {len(instruments)} Oanda instruments")
                
                return instruments
            else:
                self.logger.error(f"Failed to fetch Oanda instruments: HTTP {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching Oanda instruments: {e}")
        
        # 返回空字典如果获取失败
        return {}
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化产品代码格式"""
        symbol = symbol.upper().strip()
        
        # 移除常见的分隔符
        symbol = symbol.replace('/', '_').replace('-', '_').replace('.', '_')
        
        # 常见外汇货币对映射
        forex_mappings = {
            'EURUSD': 'EUR_USD',
            'GBPUSD': 'GBP_USD', 
            'USDJPY': 'USD_JPY',
            'USDCHF': 'USD_CHF',
            'USDCAD': 'USD_CAD',
            'AUDUSD': 'AUD_USD',
            'NZDUSD': 'NZD_USD',
            'EURGBP': 'EUR_GBP',
            'EURJPY': 'EUR_JPY',
            'GBPJPY': 'GBP_JPY'
        }
        
        # 如果是6位连续字符，转换为Oanda格式
        if len(symbol) == 6 and symbol.isalpha():
            return forex_mappings.get(symbol, f"{symbol[:3]}_{symbol[3:]}")
        
        return symbol
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.connection_status.is_connected:
                return False
            
            # 尝试获取账户信息
            response = requests.get(
                f"{self.api_url}/v3/accounts/{self.account_id}",
                headers=self.headers,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.warning(f"Oanda health check failed: {e}")
            return False