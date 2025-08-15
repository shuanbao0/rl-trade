"""
TrueFX数据源实现

提供外汇Tick级实时数据，支持：
- 16+主要货币对的实时tick数据
- 毫秒级时间戳精度
- 买卖盘价格和成交量
- 免费和认证用户访问
"""

import time
import requests
import csv
import io
import threading
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base import (
    AbstractDataSource, DataInterval, MarketType, MarketData, 
    DataSourceCapabilities, DataQuality, DataSource
)
from .converter import DataConverter


class TrueFXDataSource(AbstractDataSource):
    """TrueFX外汇数据源实现"""
    
    BASE_URL = "https://webrates.truefx.com/rates"
    
    # 支持的货币对
    UNAUTHENTICATED_PAIRS = [
        'EUR/USD', 'USD/JPY', 'GBP/USD', 'EUR/GBP', 'USD/CHF',
        'EUR/JPY', 'EUR/CHF', 'USD/CAD', 'AUD/USD', 'GBP/JPY'
    ]
    
    AUTHENTICATED_PAIRS = UNAUTHENTICATED_PAIRS + [
        'AUD/NZD', 'CAD/CHF', 'CHF/JPY', 'EUR/AUD', 'EUR/CAD'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 认证配置
        self.username = config.get('username')
        self.password = config.get('password')
        self.session_id = None
        self.authenticated = False
        
        # 请求配置
        self.timeout = config.get('timeout', 10)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # 流式数据配置
        self.stream_interval = config.get('stream_interval', 0.5)  # 更频繁的更新
        self._streaming = False
        self._streaming_thread = None
        
        # 数据格式
        self.format = config.get('format', 'csv')  # csv 或 json
        
        self.logger.info("TrueFX data source initialized")
    
    def connect(self) -> bool:
        """建立连接"""
        try:
            if self.username and self.password:
                # 认证连接
                return self._authenticate()
            else:
                # 未认证连接测试
                return self._test_unauthenticated_connection()
                
        except Exception as e:
            self.connection_status.is_connected = False
            self.connection_status.last_error = str(e)
            self.connection_status.retry_count += 1
            self.logger.error(f"TrueFX connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.stop_streaming()
        
        if self.session_id:
            try:
                # 断开认证会话
                params = {'di': self.session_id}
                requests.get(f"{self.BASE_URL}/disconnect.html", params=params, timeout=5)
                self.logger.info("TrueFX session disconnected")
            except Exception as e:
                self.logger.warning(f"Failed to disconnect TrueFX session: {e}")
            finally:
                self.session_id = None
                self.authenticated = False
        
        self.connection_status.is_connected = False
        self.logger.info("TrueFX disconnected")
    
    def _authenticate(self) -> bool:
        """建立认证连接"""
        params = {
            'u': self.username,
            'p': self.password,
            'q': 'ozrates',
            'f': self.format
        }
        
        response = requests.get(
            f"{self.BASE_URL}/connect.html", 
            params=params, 
            timeout=self.timeout
        )
        
        if response.status_code == 200:
            session_response = response.text.strip()
            if session_response and session_response != 'error':
                self.session_id = session_response
                self.authenticated = True
                self.connection_status.is_connected = True
                self.connection_status.connected_at = datetime.now()
                self.connection_status.last_error = None
                self.logger.info(f"TrueFX authenticated successfully, session: {self.session_id[:8]}...")
                return True
            else:
                # 认证失败，清除session_id
                self.session_id = None
                self.authenticated = False
        
        raise ConnectionError(f"TrueFX authentication failed: {response.text}")
    
    def _test_unauthenticated_connection(self) -> bool:
        """测试未认证连接"""
        # 尝试获取一些公开数据
        test_data = self._fetch_rates_data(['EUR/USD'])
        
        if test_data:
            self.authenticated = False
            self.connection_status.is_connected = True
            self.connection_status.connected_at = datetime.now()
            self.connection_status.last_error = None
            self.logger.info("TrueFX unauthenticated connection successful")
            return True
        else:
            raise ConnectionError("TrueFX unauthenticated connection test failed")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        注意：TrueFX主要提供实时数据，历史数据需要从下载页面获取
        """
        raise NotImplementedError(
            "TrueFX historical data requires manual download from "
            "https://www.truefx.com/truefx-historical-downloads/. "
            "Use fetch_realtime_data() for current market data."
        )
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """获取实时外汇数据"""
        start_time = time.time()
        single_symbol = isinstance(symbols, str)
        
        if single_symbol:
            symbols = [symbols]
        
        # 标准化货币对格式
        normalized_symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        
        try:
            # 获取汇率数据
            rates_data = self._fetch_rates_data(normalized_symbols)
            
            if not rates_data:
                raise ValueError(f"No data available for symbols: {symbols}")
            
            results = []
            for rate_data in rates_data:
                market_data = self._parse_rate_data(rate_data)
                if market_data:
                    results.append(market_data)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.debug(f"Fetched TrueFX data for {len(results)} symbols in {response_time:.3f}s")
            
            return results[0] if single_symbol and results else results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch TrueFX realtime data: {e}")
            raise
    
    def _fetch_rates_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """从TrueFX API获取汇率数据"""
        params = {
            'f': self.format,
            'c': ','.join(symbols)
        }
        
        # 如果有认证会话，添加会话ID
        if self.session_id:
            params['id'] = self.session_id
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.BASE_URL}/connect.html", 
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.text, symbols)
                else:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(f"TrueFX request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
        
        return []
    
    def _parse_response(self, response_text: str, requested_symbols: List[str]) -> List[Dict[str, Any]]:
        """解析TrueFX响应数据"""
        if not response_text.strip():
            return []
        
        rates_data = []
        
        try:
            # TrueFX返回CSV格式数据
            csv_reader = csv.reader(io.StringIO(response_text))
            
            for row in csv_reader:
                if len(row) >= 9:  # TrueFX标准格式：symbol,timestamp,bid,bid_points,ask,ask_points,high,low,open
                    try:
                        rate_info = {
                            'symbol': row[0],
                            'timestamp': int(row[1]) / 1000,  # 转换为秒
                            'bid': float(row[2] + row[3]) if row[2] and row[3] else None,  # bid + bid_points
                            'ask': float(row[4] + row[5]) if row[4] and row[5] else None,  # ask + ask_points
                            'high': float(row[6]) if len(row) > 6 and row[6] else None,
                            'low': float(row[7]) if len(row) > 7 and row[7] else None,
                            'open': float(row[8]) if len(row) > 8 and row[8] else None
                        }
                        rates_data.append(rate_info)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse TrueFX row {row}: {e}")
                        continue
            
        except Exception as e:
            self.logger.error(f"Failed to parse TrueFX response: {e}")
            self.logger.debug(f"Response content: {response_text[:500]}...")
        
        return rates_data
    
    def _parse_rate_data(self, rate_data: Dict[str, Any]) -> Optional[MarketData]:
        """将TrueFX汇率数据转换为MarketData"""
        try:
            symbol = rate_data['symbol']
            timestamp = datetime.fromtimestamp(rate_data['timestamp'])
            
            bid = rate_data.get('bid')
            ask = rate_data.get('ask')
            
            # TrueFX没有直接的OHLC，使用买卖中价估算
            mid_price = (bid + ask) / 2 if bid and ask else None
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=rate_data.get('open') or mid_price or 0,
                high=rate_data.get('high') or mid_price or 0,
                low=rate_data.get('low') or mid_price or 0,
                close=mid_price or 0,
                volume=0,  # TrueFX不提供成交量
                bid=bid,
                ask=ask,
                spread=ask - bid if bid and ask else None,
                source='truefx',
                quality=DataQuality.HIGH,  # TrueFX提供高质量tick数据
                metadata={
                    'tick_time': rate_data['timestamp'],
                    'authenticated': self.authenticated
                }
            )
            
            return market_data
            
        except Exception as e:
            self.logger.warning(f"Failed to parse rate data {rate_data}: {e}")
            return None
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable,
        interval_seconds: float = None
    ) -> None:
        """流式接收实时数据"""
        if self._streaming:
            self.logger.warning("TrueFX streaming already active")
            return
        
        interval_seconds = interval_seconds or self.stream_interval
        normalized_symbols = [self._normalize_symbol(symbol) for symbol in symbols]
        
        def stream_worker():
            """流式数据工作线程"""
            self._streaming = True
            self.logger.info(f"Started TrueFX streaming for {len(symbols)} pairs, interval: {interval_seconds}s")
            
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self._streaming:
                try:
                    # 获取实时数据
                    data = self.fetch_realtime_data(normalized_symbols)
                    if data:
                        callback(data)
                        consecutive_errors = 0  # 重置错误计数
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.warning(f"TrueFX streaming error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive TrueFX streaming errors, stopping")
                        break
                    
                    # 错误时使用指数退避
                    error_delay = min(interval_seconds * (2 ** consecutive_errors), 30)
                    time.sleep(error_delay)
            
            self._streaming = False
            self.logger.info("TrueFX streaming stopped")
        
        # 启动流式数据线程
        self._streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self._streaming_thread.start()
    
    def stop_streaming(self) -> None:
        """停止流式数据接收"""
        if self._streaming:
            self._streaming = False
            if self._streaming_thread and self._streaming_thread.is_alive():
                self._streaming_thread.join(timeout=5)
            self.logger.info("TrueFX streaming stopped")
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证货币对是否支持"""
        normalized_symbol = self._normalize_symbol(symbol)
        
        if self.authenticated:
            return normalized_symbol in self.AUTHENTICATED_PAIRS
        else:
            return normalized_symbol in self.UNAUTHENTICATED_PAIRS
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """搜索货币对"""
        available_pairs = self.AUTHENTICATED_PAIRS if self.authenticated else self.UNAUTHENTICATED_PAIRS
        
        query_upper = query.upper()
        results = []
        
        for pair in available_pairs:
            if query_upper in pair:
                results.append({
                    'symbol': pair,
                    'name': f"{pair} Spot Rate",
                    'type': 'forex',
                    'source': 'truefx'
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="TrueFX",
            source_id=DataSource.TRUEFX,
            supported_markets=[MarketType.FOREX],
            supported_intervals=[DataInterval.TICK],
            has_realtime=True,
            has_historical=False,  # 需要手动下载
            has_streaming=True,
            requires_auth=False,  # 可选认证
            is_free=True,
            max_history_days=None,
            min_interval=DataInterval.TICK,
            max_symbols_per_request=15,  # TrueFX支持最多15个货币对
            rate_limits={},  # TrueFX没有明确的速率限制
            data_quality=DataQuality.HIGH,
            latency_ms=500,  # 毫秒级延迟
            api_version="TrueFX API v1",
            documentation_url="https://www.truefx.com/dev/data-feed-api/",
            support_contact="support@truefx.com"
        )
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化货币对格式"""
        symbol = symbol.upper().strip()
        
        # 移除常见的分隔符并转换为TrueFX格式
        symbol = symbol.replace('_', '').replace('-', '').replace('.', '')
        
        # 确保是6位货币对格式并添加斜杠
        if len(symbol) == 6:
            return f"{symbol[:3]}/{symbol[3:]}"
        elif len(symbol) == 7 and '/' in symbol:
            return symbol
        else:
            # 常见货币对映射
            currency_mappings = {
                'EURUSD': 'EUR/USD',
                'GBPUSD': 'GBP/USD',
                'USDJPY': 'USD/JPY',
                'EURGBP': 'EUR/GBP',
                'USDCHF': 'USD/CHF',
                'EURJPY': 'EUR/JPY',
                'EURCHF': 'EUR/CHF',
                'USDCAD': 'USD/CAD',
                'AUDUSD': 'AUD/USD',
                'GBPJPY': 'GBP/JPY',
                'AUDNZD': 'AUD/NZD',
                'CADCHF': 'CAD/CHF',
                'CHFJPY': 'CHF/JPY',
                'EURAUD': 'EUR/AUD',
                'EURCAD': 'EUR/CAD'
            }
            
            return currency_mappings.get(symbol, symbol)
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.connection_status.is_connected:
                return False
            
            # 尝试获取EUR/USD数据
            test_data = self.fetch_realtime_data('EUR/USD')
            return test_data is not None
            
        except Exception as e:
            self.logger.warning(f"TrueFX health check failed: {e}")
            return False