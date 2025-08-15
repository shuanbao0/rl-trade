"""
HistData数据源实现

提供历史外汇Tick和分钟数据，特点：
- 2000年起的丰富历史数据
- 支持Tick级和1分钟OHLC数据
- 免费数据，需手动下载CSV文件
- 支持多种文件格式和时区处理
"""

import os
import time
import zipfile
import requests
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from .base import (
    AbstractDataSource, DataInterval, MarketType, MarketData, 
    DataSourceCapabilities, DataQuality, DataSource
)
from .converter import DataConverter


class HistDataDataSource(AbstractDataSource):
    """HistData历史数据源实现"""
    
    BASE_URL = "https://www.histdata.com"
    
    # 支持的货币对（HistData主要货币对）
    SUPPORTED_PAIRS = [
        'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 
        'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDCAD',
        'USDCHF', 'USDJPY', 'XAUUSD', 'XAGUSD'  # 包含黄金白银
    ]
    
    # 数据格式类型
    DATA_FORMATS = {
        'generic_ascii': 'Generic ASCII',
        'metatrader': 'MetaTrader',
        'excel': 'Microsoft Excel',
        'ninjatrader': 'NinjaTrader',
        'metastock': 'MetaStock'
    }
    
    # 时间间隔映射
    INTERVAL_MAP = {
        DataInterval.TICK: 'tick',
        DataInterval.MINUTE_1: 'M1'
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 数据存储配置
        self.data_directory = Path(config.get('data_directory', 'histdata_cache'))
        self.data_directory.mkdir(exist_ok=True)
        
        # 下载配置
        self.auto_download = config.get('auto_download', False)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        
        # 数据格式配置
        self.default_format = config.get('format', 'generic_ascii')
        self.timezone_offset = config.get('timezone_offset', 0)  # EST偏移
        
        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # 数据缓存
        self._available_data = {}
        self._cache_ttl = 3600  # 1小时缓存
        self._cache_time = None
        
        self.logger.info(f"HistData source initialized with directory: {self.data_directory}")
    
    def connect(self) -> bool:
        """建立连接（检查数据目录和可用性）"""
        try:
            # 检查数据目录是否可写
            test_file = self.data_directory / '.test'
            test_file.touch()
            test_file.unlink()
            
            # 尝试访问HistData网站（如果启用自动下载）
            if self.auto_download:
                response = requests.head(self.BASE_URL, timeout=10, headers=self.headers)
                if response.status_code != 200:
                    raise ConnectionError(f"HistData website unreachable: {response.status_code}")
            
            self.connection_status.is_connected = True
            self.connection_status.connected_at = datetime.now()
            self.connection_status.last_error = None
            
            # 扫描已有数据文件
            self._scan_available_data()
            
            self.logger.info("HistData connection successful")
            return True
            
        except Exception as e:
            self.connection_status.is_connected = False
            self.connection_status.last_error = str(e)
            self.connection_status.retry_count += 1
            self.logger.error(f"HistData connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.connection_status.is_connected = False
        self.logger.info("HistData disconnected")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: DataInterval
    ) -> pd.DataFrame:
        """获取历史数据"""
        start_time = time.time()
        
        if interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {list(self.INTERVAL_MAP.keys())}")
        
        normalized_symbol = self._normalize_symbol(symbol)
        interval_type = self.INTERVAL_MAP[interval]
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        try:
            # 获取数据文件
            data_files = self._get_data_files(normalized_symbol, start_date, end_date, interval_type)
            
            if not data_files:
                if self.auto_download:
                    # 尝试自动下载
                    self._download_data(normalized_symbol, start_date, end_date, interval_type)
                    data_files = self._get_data_files(normalized_symbol, start_date, end_date, interval_type)
                
                if not data_files:
                    raise FileNotFoundError(
                        f"No data files found for {symbol} from {start_date.date()} to {end_date.date()}. "
                        f"Please download files manually from histdata.com or enable auto_download."
                    )
            
            # 读取和合并数据文件
            all_data = []
            for file_path in data_files:
                df = self._read_data_file(file_path, interval_type)
                if df is not None and not df.empty:
                    # 过滤时间范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    if not df.empty:
                        all_data.append(df)
            
            if not all_data:
                raise ValueError(f"No data found in the specified time range")
            
            # 合并所有数据
            result_df = pd.concat(all_data, axis=0)
            result_df.sort_index(inplace=True)
            result_df.drop_duplicates(inplace=True)
            
            # 添加元数据
            result_df['symbol'] = normalized_symbol
            result_df['source'] = 'histdata'
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.info(f"Loaded HistData for {symbol}: {len(result_df)} records in {response_time:.3f}s")
            
            return result_df
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch HistData: {e}")
            raise
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """
        获取实时数据
        
        注意：HistData主要提供历史数据，实时数据功能有限
        这里返回最新的历史数据作为"实时"数据
        """
        single_symbol = isinstance(symbols, str)
        if single_symbol:
            symbols = [symbols]
        
        results = []
        for symbol in symbols:
            try:
                normalized_symbol = self._normalize_symbol(symbol)
                
                # 获取最近的数据文件
                latest_file = self._get_latest_data_file(normalized_symbol)
                if latest_file:
                    df = self._read_data_file(latest_file, 'M1')
                    if df is not None and not df.empty:
                        # 获取最后一条记录
                        last_row = df.iloc[-1]
                        
                        market_data = MarketData(
                            symbol=normalized_symbol,
                            timestamp=df.index[-1].to_pydatetime(),
                            open=last_row['open'],
                            high=last_row['high'],
                            low=last_row['low'],
                            close=last_row['close'],
                            volume=last_row.get('volume', 0),
                            source='histdata',
                            quality=DataQuality.MEDIUM,  # 历史数据质量
                            metadata={'is_historical': True, 'file_path': str(latest_file)}
                        )
                        results.append(market_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to get latest data for {symbol}: {e}")
                continue
        
        if not results:
            raise ValueError(f"No recent data available for symbols: {symbols}")
        
        return results[0] if single_symbol else results
    
    def _scan_available_data(self):
        """扫描数据目录中的可用文件"""
        self._available_data = {}
        
        for file_path in self.data_directory.rglob('*.csv'):
            try:
                # 解析文件名获取货币对和日期信息
                file_info = self._parse_filename(file_path)
                if file_info:
                    symbol = file_info['symbol']
                    if symbol not in self._available_data:
                        self._available_data[symbol] = []
                    self._available_data[symbol].append({
                        'path': file_path,
                        'info': file_info
                    })
            except Exception as e:
                self.logger.debug(f"Failed to parse file {file_path}: {e}")
        
        self._cache_time = time.time()
        self.logger.info(f"Found data for {len(self._available_data)} symbols")
    
    def _parse_filename(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析HistData文件名获取信息"""
        filename = file_path.stem
        
        # HistData文件名格式通常为：SYMBOL_YEAR_MONTH.csv 或 SYMBOL_TICK_YYYY_MM.csv
        parts = filename.upper().split('_')
        
        if len(parts) >= 3:
            symbol = parts[0]
            if symbol in self.SUPPORTED_PAIRS:
                # 检查是否为Tick数据
                if 'TICK' in parts:
                    data_type = 'tick'
                    year_idx = parts.index('TICK') + 1
                else:
                    data_type = 'M1'
                    year_idx = 1
                
                try:
                    year = int(parts[year_idx])
                    month = int(parts[year_idx + 1]) if len(parts) > year_idx + 1 else 1
                    
                    return {
                        'symbol': symbol,
                        'data_type': data_type,
                        'year': year,
                        'month': month,
                        'date': datetime(year, month, 1)
                    }
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def _get_data_files(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        interval_type: str
    ) -> List[Path]:
        """获取指定时间范围的数据文件"""
        if symbol not in self._available_data:
            return []
        
        matching_files = []
        for file_info in self._available_data[symbol]:
            info = file_info['info']
            if info['data_type'] == interval_type:
                file_date = info['date']
                # 检查文件日期是否在范围内
                file_end = datetime(file_date.year, file_date.month + 1, 1) - timedelta(days=1)
                if not (file_end < start_date or file_date > end_date):
                    matching_files.append(file_info['path'])
        
        return sorted(matching_files)
    
    def _get_latest_data_file(self, symbol: str) -> Optional[Path]:
        """获取最新的数据文件"""
        if symbol not in self._available_data:
            return None
        
        latest_file = None
        latest_date = None
        
        for file_info in self._available_data[symbol]:
            file_date = file_info['info']['date']
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = file_info['path']
        
        return latest_file
    
    def _read_data_file(self, file_path: Path, interval_type: str) -> Optional[pd.DataFrame]:
        """读取HistData CSV文件"""
        try:
            if interval_type == 'tick':
                return self._read_tick_file(file_path)
            elif interval_type == 'M1':
                return self._read_m1_file(file_path)
            else:
                raise ValueError(f"Unsupported interval type: {interval_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to read data file {file_path}: {e}")
            return None
    
    def _read_tick_file(self, file_path: Path) -> pd.DataFrame:
        """读取Tick数据文件"""
        # HistData Tick格式: YYYYMMDD HHMMSS,BID,ASK
        df = pd.read_csv(
            file_path,
            names=['datetime', 'bid', 'ask'],
            parse_dates=['datetime'],
            date_format='%Y%m%d %H%M%S'
        )
        
        # 计算中间价和价差
        df['close'] = (df['bid'] + df['ask']) / 2
        df['open'] = df['close']  # Tick数据没有开盘价概念
        df['high'] = df['close']
        df['low'] = df['close']
        df['volume'] = 0  # Tick数据通常没有成交量
        df['spread'] = df['ask'] - df['bid']
        
        # 设置时间索引
        df.set_index('datetime', inplace=True)
        
        return df
    
    def _read_m1_file(self, file_path: Path) -> pd.DataFrame:
        """读取1分钟OHLC数据文件"""
        # HistData M1格式: YYYYMMDD HHMMSS,OPEN,HIGH,LOW,CLOSE,VOLUME
        df = pd.read_csv(
            file_path,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['datetime'],
            date_format='%Y%m%d %H%M%S'
        )
        
        # 设置时间索引
        df.set_index('datetime', inplace=True)
        
        # 数据清洗
        df = df.dropna()
        df = df[df['volume'] >= 0]  # 移除负成交量
        
        return df
    
    def _download_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime, 
        interval_type: str
    ):
        """自动下载数据（如果启用）"""
        if not self.auto_download:
            return
        
        self.logger.info(f"Attempting to download {symbol} data for {start_date.date()} to {end_date.date()}")
        
        # 这里可以实现自动下载逻辑
        # 由于HistData.com的下载通常需要手动操作，这里主要是占位符
        # 实际实现可能需要selenium等工具进行网页自动化
        
        raise NotImplementedError(
            "Automatic download from HistData is not implemented. "
            "Please download data manually from https://www.histdata.com/download-free-forex-data/"
        )
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证货币对是否支持"""
        normalized_symbol = self._normalize_symbol(symbol)
        return normalized_symbol in self.SUPPORTED_PAIRS
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """搜索货币对"""
        query_upper = query.upper()
        results = []
        
        for pair in self.SUPPORTED_PAIRS:
            if query_upper in pair:
                # 格式化显示名称 - 特殊商品优先处理
                if pair == 'XAUUSD':
                    display_name = "Gold Spot"
                    market_type = 'commodity'
                elif pair == 'XAGUSD':
                    display_name = "Silver Spot"
                    market_type = 'commodity'
                elif len(pair) == 6 and pair.isalpha():
                    display_name = f"{pair[:3]}/{pair[3:]}"
                    market_type = 'forex'
                else:
                    display_name = pair
                    market_type = 'forex'
                
                results.append({
                    'symbol': pair,
                    'name': display_name,
                    'type': market_type,
                    'source': 'histdata'
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="HistData",
            source_id=DataSource.HISTDATA,
            supported_markets=[MarketType.FOREX, MarketType.COMMODITIES],
            supported_intervals=[DataInterval.TICK, DataInterval.MINUTE_1],
            has_realtime=False,  # 主要是历史数据
            has_historical=True,
            has_streaming=False,
            requires_auth=False,
            is_free=True,
            max_history_days=9000,  # 从2000年开始
            min_interval=DataInterval.TICK,
            max_symbols_per_request=1,  # 文件基础，一次处理一个
            rate_limits={},
            data_quality=DataQuality.HIGH,
            latency_ms=None,  # 文件读取，无网络延迟
            api_version="File-based CSV",
            documentation_url="https://www.histdata.com/f-a-q/data-files-detailed-specification/",
            support_contact="support@histdata.com"
        )
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化货币对格式"""
        symbol = symbol.upper().strip()
        
        # 移除常见的分隔符
        symbol = symbol.replace('/', '').replace('-', '').replace('_', '').replace('.', '')
        
        # 确保是HistData支持的格式
        if len(symbol) == 6 and symbol.isalpha():
            # 检查是否在支持列表中
            if symbol in self.SUPPORTED_PAIRS:
                return symbol
            
            # 尝试常见的货币对映射
            if symbol == 'XAUUSD':
                return 'XAUUSD'
            elif symbol == 'XAGUSD':
                return 'XAGUSD'
        
        return symbol
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.connection_status.is_connected:
                return False
            
            # 检查数据目录是否可访问
            return self.data_directory.exists() and self.data_directory.is_dir()
            
        except Exception as e:
            self.logger.warning(f"HistData health check failed: {e}")
            return False