"""
YFinance数据源实现

基于现有DataManager的功能，适配到新的抽象数据源接口
支持股票、外汇和加密货币数据
"""

import time
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import threading

from .base import (
    AbstractDataSource, DataInterval, MarketType, MarketData, 
    DataSourceCapabilities, DataQuality
)
from .converter import DataConverter


class YFinanceDataSource(AbstractDataSource):
    """YFinance数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 代理设置
        self.proxy = config.get('proxy')
        if self.proxy:
            self._setup_proxy()
        
        # 流式数据相关
        self._streaming = False
        self._streaming_thread = None
        
        # YFinance特定配置
        self.auto_adjust_interval = config.get('auto_adjust_interval', True)
        self.poll_interval = config.get('poll_interval', 5.0)  # 实时数据轮询间隔
        
        self.logger.info("YFinance data source initialized")
    
    def _setup_proxy(self) -> None:
        """设置代理"""
        try:
            if hasattr(yf, 'set_config'):
                yf.set_config(proxy=self.proxy)
                self.logger.info(f"Set YFinance proxy: {self.proxy}")
            else:
                import os
                os.environ['HTTP_PROXY'] = self.proxy
                os.environ['HTTPS_PROXY'] = self.proxy
                os.environ['http_proxy'] = self.proxy
                os.environ['https_proxy'] = self.proxy
                self.logger.info(f"Set proxy via environment variables: {self.proxy}")
        except Exception as e:
            self.logger.warning(f"Failed to set proxy: {e}")
    
    def connect(self) -> bool:
        """建立连接（YFinance无需显式连接）"""
        try:
            # 测试连接：尝试获取一个常见股票的信息
            test_ticker = yf.Ticker("AAPL")
            info = test_ticker.info
            
            if info and 'symbol' in info:
                self.connection_status.is_connected = True
                self.connection_status.connected_at = datetime.now()
                self.connection_status.last_error = None
                self.connection_status.retry_count = 0
                self.logger.info("YFinance connection test successful")
                return True
            else:
                raise ConnectionError("YFinance connection test failed")
                
        except Exception as e:
            self.connection_status.is_connected = False
            self.connection_status.last_error = str(e)
            self.connection_status.retry_count += 1
            self.logger.error(f"YFinance connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.stop_streaming()
        self.connection_status.is_connected = False
        self.logger.info("YFinance disconnected")
    
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
        start_time = time.time()
        
        try:
            # 转换日期格式
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # 转换时间间隔
            yf_interval = self._convert_interval_to_yfinance(interval)
            
            # 直接使用yfinance API获取数据
            symbol_normalized = self._normalize_symbol(symbol)
            ticker = yf.Ticker(symbol_normalized)
            
            # 使用日期范围获取数据
            history_kwargs = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'interval': yf_interval
            }
            
            # 添加代理支持
            if self.proxy:
                history_kwargs['proxy'] = self.proxy
                
            data = ticker.history(**history_kwargs)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # 标准化DataFrame
            data = DataConverter.standardize_columns(data)
            data = DataConverter.clean_data(data)
            
            # 添加数据源信息
            data['source'] = 'yfinance'
            data['symbol'] = symbol_normalized
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.info(
                f"Fetched {len(data)} records for {symbol} "
                f"({interval.value}) in {response_time:.2f}s"
            )
            
            return data
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            raise
    
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
        start_time = time.time()
        single_symbol = isinstance(symbols, str)
        
        if single_symbol:
            symbols = [symbols]
        
        results = []
        
        try:
            for symbol in symbols:
                symbol_normalized = self._normalize_symbol(symbol)
                
                # 获取实时数据
                ticker = yf.Ticker(symbol_normalized)
                info = ticker.info
                
                if not info:
                    raise ValueError(f"No data available for {symbol}")
                
                # 构建MarketData
                current_time = datetime.now()
                
                # 提取价格信息
                current_price = (
                    info.get('regularMarketPrice') or 
                    info.get('currentPrice') or
                    info.get('previousClose', 0)
                )
                
                open_price = (
                    info.get('regularMarketOpen') or 
                    info.get('open') or
                    current_price
                )
                
                high_price = (
                    info.get('regularMarketDayHigh') or
                    info.get('dayHigh') or
                    max(open_price, current_price)
                )
                
                low_price = (
                    info.get('regularMarketDayLow') or
                    info.get('dayLow') or
                    min(open_price, current_price)
                )
                
                volume = (
                    info.get('regularMarketVolume') or
                    info.get('volume') or
                    0
                )
                
                # 外汇特有数据
                bid = info.get('bid')
                ask = info.get('ask')
                
                market_data = MarketData(
                    symbol=symbol_normalized,
                    timestamp=current_time,
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(current_price),
                    volume=float(volume),
                    bid=float(bid) if bid else None,
                    ask=float(ask) if ask else None,
                    spread=float(ask - bid) if (bid and ask) else None,
                    source='yfinance',
                    quality=DataQuality.MEDIUM,  # YFinance为中等质量
                    metadata={
                        'market_cap': info.get('marketCap'),
                        'currency': info.get('currency'),
                        'exchange': info.get('exchange'),
                        'sector': info.get('sector'),
                        'industry': info.get('industry')
                    }
                )
                
                results.append(market_data)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.debug(f"Fetched realtime data for {len(symbols)} symbols")
            
            return results[0] if single_symbol else results
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch realtime data: {e}")
            raise
    
    def stream_realtime_data(
        self,
        symbols: List[str],
        callback: callable,
        interval_seconds: float = None
    ) -> None:
        """
        流式接收实时数据（使用轮询实现）
        
        Args:
            symbols: 标的代码列表
            callback: 数据回调函数
            interval_seconds: 轮询间隔（秒），默认使用配置的poll_interval
        """
        if self._streaming:
            self.logger.warning("Streaming already active")
            return
        
        interval_seconds = interval_seconds or self.poll_interval
        
        def stream_worker():
            """流式数据工作线程"""
            self._streaming = True
            self.logger.info(f"Started streaming {len(symbols)} symbols, interval: {interval_seconds}s")
            
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self._streaming:
                try:
                    # 获取实时数据
                    data = self.fetch_realtime_data(symbols)
                    callback(data)
                    
                    consecutive_errors = 0  # 重置错误计数
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.warning(f"Streaming error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors, stopping stream")
                        break
                    
                    # 错误时使用更长的等待时间
                    error_delay = min(interval_seconds * consecutive_errors, 60)
                    time.sleep(error_delay)
            
            self._streaming = False
            self.logger.info("Streaming stopped")
        
        # 启动流式数据线程
        self._streaming_thread = threading.Thread(target=stream_worker, daemon=True)
        self._streaming_thread.start()
    
    def stop_streaming(self) -> None:
        """停止流式数据接收"""
        if self._streaming:
            self._streaming = False
            if self._streaming_thread and self._streaming_thread.is_alive():
                self._streaming_thread.join(timeout=5)
            self.logger.info("Streaming stopped")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证标的代码
        
        Args:
            symbol: 标的代码
            
        Returns:
            是否有效
        """
        try:
            symbol_normalized = self._normalize_symbol(symbol)
            ticker = yf.Ticker(symbol_normalized)
            info = ticker.info
            
            # 检查是否有有效的数据
            return bool(info and ('symbol' in info or 'shortName' in info))
            
        except Exception as e:
            self.logger.debug(f"Symbol validation failed for {symbol}: {e}")
            return False
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        搜索标的（YFinance无直接搜索API，返回常见匹配）
        
        Args:
            query: 搜索关键词
            limit: 最大返回数量
            
        Returns:
            匹配的标的列表
        """
        # YFinance没有搜索API，提供一些常见标的的启发式匹配
        common_symbols = [
            # 美股
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'type': 'stock'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'type': 'stock'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'type': 'stock'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'type': 'stock'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'type': 'stock'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'type': 'stock'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'type': 'stock'},
            
            # 外汇
            {'symbol': 'EURUSD=X', 'name': 'EUR/USD', 'type': 'forex'},
            {'symbol': 'GBPUSD=X', 'name': 'GBP/USD', 'type': 'forex'},
            {'symbol': 'USDJPY=X', 'name': 'USD/JPY', 'type': 'forex'},
            {'symbol': 'USDCHF=X', 'name': 'USD/CHF', 'type': 'forex'},
            
            # 加密货币
            {'symbol': 'BTC-USD', 'name': 'Bitcoin USD', 'type': 'crypto'},
            {'symbol': 'ETH-USD', 'name': 'Ethereum USD', 'type': 'crypto'},
            
            # 指数
            {'symbol': '^GSPC', 'name': 'S&P 500', 'type': 'index'},
            {'symbol': '^DJI', 'name': 'Dow Jones', 'type': 'index'},
            {'symbol': '^IXIC', 'name': 'NASDAQ', 'type': 'index'},
        ]
        
        query_upper = query.upper()
        results = []
        
        for item in common_symbols:
            if (query_upper in item['symbol'] or 
                query_upper in item['name'].upper()):
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="YFinance",
            supported_markets=[
                MarketType.STOCK,
                MarketType.FOREX,
                MarketType.CRYPTO,
                MarketType.ETF,
                MarketType.INDEX,
                MarketType.FUTURES,
                MarketType.OPTIONS
            ],
            supported_intervals=[
                DataInterval.MINUTE_1,
                DataInterval.MINUTE_2,
                DataInterval.MINUTE_5,
                DataInterval.MINUTE_15,
                DataInterval.MINUTE_30,
                DataInterval.MINUTE_90,
                DataInterval.HOUR_1,
                DataInterval.DAY_1,
                DataInterval.WEEK_1,
                DataInterval.MONTH_1,
                DataInterval.MONTH_3
            ],
            has_realtime=True,
            has_historical=True,
            has_streaming=True,  # 通过轮询实现
            requires_auth=False,
            is_free=True,
            max_history_days=None,  # 理论上无限制
            min_interval=DataInterval.MINUTE_1,
            max_symbols_per_request=10,  # 建议限制
            rate_limits={
                "requests_per_minute": 2000,
                "historical_requests_per_hour": 500
            },
            data_quality=DataQuality.MEDIUM,
            latency_ms=2000,  # 约2秒延迟
            api_version="yfinance-0.2.x",
            documentation_url="https://github.com/ranaroussi/yfinance",
            support_contact="https://github.com/ranaroussi/yfinance/issues"
        )
    
    def _convert_interval_to_yfinance(self, interval: DataInterval) -> str:
        """转换时间间隔为YFinance格式"""
        interval_mapping = {
            DataInterval.MINUTE_1: "1m",
            DataInterval.MINUTE_2: "2m", 
            DataInterval.MINUTE_5: "5m",
            DataInterval.MINUTE_15: "15m",
            DataInterval.MINUTE_30: "30m",
            DataInterval.MINUTE_90: "90m",
            DataInterval.HOUR_1: "1h",
            DataInterval.DAY_1: "1d",
            DataInterval.WEEK_1: "1wk",
            DataInterval.MONTH_1: "1mo",
            DataInterval.MONTH_3: "3mo"
        }
        
        yf_interval = interval_mapping.get(interval)
        if yf_interval is None:
            raise ValueError(f"Unsupported interval for YFinance: {interval}")
            
        return yf_interval
    
    def _days_to_period_string(self, days: int) -> str:
        """将天数转换为YFinance period字符串"""
        if days <= 7:
            return "7d"
        elif days <= 30:
            return "1mo"
        elif days <= 90:
            return "3mo"
        elif days <= 180:
            return "6mo"
        elif days <= 365:
            return "1y"
        elif days <= 365 * 2:
            return "2y"
        elif days <= 365 * 5:
            return "5y"
        elif days <= 365 * 10:
            return "10y"
        else:
            return "max"
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码
        
        Args:
            symbol: 原始股票代码
            
        Returns:
            标准化后的股票代码
        """
        symbol = symbol.upper().strip()
        
        # 外汇对转换
        forex_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            'EURGBP': 'EURGBP=X',
            'EURJPY': 'EURJPY=X',
            'EURCHF': 'EURCHF=X'
        }
        
        # 加密货币转换
        crypto_mapping = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'LTC': 'LTC-USD',
            'BCH': 'BCH-USD',
            'ADA': 'ADA-USD',
            'DOT': 'DOT-USD',
            'LINK': 'LINK-USD'
        }
        
        # 应用映射
        if symbol in forex_mapping:
            return forex_mapping[symbol]
        elif symbol in crypto_mapping:
            return crypto_mapping[symbol]
        else:
            return symbol