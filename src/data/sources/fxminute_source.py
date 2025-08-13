"""
FX-1-Minute-Data 本地数据源实现

基于已下载的 FX-1-Minute-Data 数据集
支持66+个外汇对、商品和股指的高质量历史数据
直接读取本地ZIP文件，无需网络连接
"""

import os
import zipfile
import time
import csv
from typing import Union, List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from .base import (
    AbstractDataSource, DataInterval, MarketType, MarketData, 
    DataSourceCapabilities, DataQuality
)
from .converter import DataConverter


class FXMinuteDataSource(AbstractDataSource):
    """FX-1-Minute-Data 本地数据源实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 基础配置
        self.data_directory = Path(config.get('data_directory', 'data_cache/FX-1-Minute-Data'))
        self.auto_extract = config.get('auto_extract', True)
        self.cache_extracted = config.get('cache_extracted', True)
        self.extracted_cache_dir = Path(config.get('extracted_cache_dir', 'fx_minute_cache'))
        
        # 确保目录存在
        self.extracted_cache_dir.mkdir(exist_ok=True)
        
        # 支持的交易对映射（基于实际数据集结构）
        self.supported_pairs = self._load_supported_pairs()
        
        # 数据文件索引缓存
        self._file_index = {}
        self._index_loaded = False
        
        self.logger.info(f"FXMinuteData source initialized with directory: {self.data_directory}")
    
    def _load_supported_pairs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载支持的交易对列表
        基于FX-1-Minute-Data项目的完整数据集
        """
        return {
            # 主要货币对
            'eurusd': {'name': 'EUR/USD', 'type': 'forex', 'start_year': 2000},
            'gbpusd': {'name': 'GBP/USD', 'type': 'forex', 'start_year': 2000},
            'usdjpy': {'name': 'USD/JPY', 'type': 'forex', 'start_year': 2000},
            'usdchf': {'name': 'USD/CHF', 'type': 'forex', 'start_year': 2000},
            'audusd': {'name': 'AUD/USD', 'type': 'forex', 'start_year': 2000},
            'usdcad': {'name': 'USD/CAD', 'type': 'forex', 'start_year': 2000},
            'nzdusd': {'name': 'NZD/USD', 'type': 'forex', 'start_year': 2005},
            
            # 交叉货币对
            'eurgbp': {'name': 'EUR/GBP', 'type': 'forex', 'start_year': 2002},
            'eurjpy': {'name': 'EUR/JPY', 'type': 'forex', 'start_year': 2002},
            'eurchf': {'name': 'EUR/CHF', 'type': 'forex', 'start_year': 2002},
            'euraud': {'name': 'EUR/AUD', 'type': 'forex', 'start_year': 2002},
            'eurcad': {'name': 'EUR/CAD', 'type': 'forex', 'start_year': 2007},
            'eurnzd': {'name': 'EUR/NZD', 'type': 'forex', 'start_year': 2008},
            'gbpjpy': {'name': 'GBP/JPY', 'type': 'forex', 'start_year': 2002},
            'gbpchf': {'name': 'GBP/CHF', 'type': 'forex', 'start_year': 2002},
            'gbpaud': {'name': 'GBP/AUD', 'type': 'forex', 'start_year': 2007},
            'gbpcad': {'name': 'GBP/CAD', 'type': 'forex', 'start_year': 2007},
            'gbpnzd': {'name': 'GBP/NZD', 'type': 'forex', 'start_year': 2008},
            'audjpy': {'name': 'AUD/JPY', 'type': 'forex', 'start_year': 2002},
            'audchf': {'name': 'AUD/CHF', 'type': 'forex', 'start_year': 2008},
            'audcad': {'name': 'AUD/CAD', 'type': 'forex', 'start_year': 2007},
            'audnzd': {'name': 'AUD/NZD', 'type': 'forex', 'start_year': 2007},
            'nzdjpy': {'name': 'NZD/JPY', 'type': 'forex', 'start_year': 2006},
            'nzdchf': {'name': 'NZD/CHF', 'type': 'forex', 'start_year': 2008},
            'nzdcad': {'name': 'NZD/CAD', 'type': 'forex', 'start_year': 2008},
            'chfjpy': {'name': 'CHF/JPY', 'type': 'forex', 'start_year': 2002},
            'cadchf': {'name': 'CAD/CHF', 'type': 'forex', 'start_year': 2008},
            'cadjpy': {'name': 'CAD/JPY', 'type': 'forex', 'start_year': 2007},
            
            # 新兴市场货币
            'usdmxn': {'name': 'USD/MXN', 'type': 'forex', 'start_year': 2010},
            'usdtry': {'name': 'USD/TRY', 'type': 'forex', 'start_year': 2010},
            'usdzar': {'name': 'USD/ZAR', 'type': 'forex', 'start_year': 2010},
            'usdhuf': {'name': 'USD/HUF', 'type': 'forex', 'start_year': 2010},
            'usdpln': {'name': 'USD/PLN', 'type': 'forex', 'start_year': 2010},
            'usdczk': {'name': 'USD/CZK', 'type': 'forex', 'start_year': 2010},
            'eurtry': {'name': 'EUR/TRY', 'type': 'forex', 'start_year': 2010},
            'eurhuf': {'name': 'EUR/HUF', 'type': 'forex', 'start_year': 2010},
            'eurpln': {'name': 'EUR/PLN', 'type': 'forex', 'start_year': 2010},
            'eurczk': {'name': 'EUR/CZK', 'type': 'forex', 'start_year': 2010},
            'zarjpy': {'name': 'ZAR/JPY', 'type': 'forex', 'start_year': 2010},
            
            # 北欧货币
            'eurnok': {'name': 'EUR/NOK', 'type': 'forex', 'start_year': 2008},
            'eursek': {'name': 'EUR/SEK', 'type': 'forex', 'start_year': 2008},
            'eurdkk': {'name': 'EUR/DKK', 'type': 'forex', 'start_year': 2008},
            'usdnok': {'name': 'USD/NOK', 'type': 'forex', 'start_year': 2008},
            'usdsek': {'name': 'USD/SEK', 'type': 'forex', 'start_year': 2008},
            'usddkk': {'name': 'USD/DKK', 'type': 'forex', 'start_year': 2008},
            
            # 亚洲货币
            'usdhkd': {'name': 'USD/HKD', 'type': 'forex', 'start_year': 2008},
            'usdsgd': {'name': 'USD/SGD', 'type': 'forex', 'start_year': 2008},
            'sgdjpy': {'name': 'SGD/JPY', 'type': 'forex', 'start_year': 2008},
            
            # 商品
            'xauusd': {'name': 'XAU/USD (Gold)', 'type': 'commodity', 'start_year': 2009},
            'xagusd': {'name': 'XAG/USD (Silver)', 'type': 'commodity', 'start_year': 2009},
            'xaueur': {'name': 'XAU/EUR (Gold in EUR)', 'type': 'commodity', 'start_year': 2009},
            'xaugbp': {'name': 'XAU/GBP (Gold in GBP)', 'type': 'commodity', 'start_year': 2009},
            'xauchf': {'name': 'XAU/CHF (Gold in CHF)', 'type': 'commodity', 'start_year': 2009},
            'xauaud': {'name': 'XAU/AUD (Gold in AUD)', 'type': 'commodity', 'start_year': 2009},
            'wtiusd': {'name': 'WTI/USD (WTI Oil)', 'type': 'commodity', 'start_year': 2010},
            'bcousd': {'name': 'BCO/USD (Brent Oil)', 'type': 'commodity', 'start_year': 2010},
            
            # 股指
            'spxusd': {'name': 'SPX/USD (S&P 500)', 'type': 'index', 'start_year': 2010},
            'jpxjpy': {'name': 'JPX/JPY (Nikkei 225)', 'type': 'index', 'start_year': 2010},
            'nsxusd': {'name': 'NSX/USD (NASDAQ 100)', 'type': 'index', 'start_year': 2010},
            'frxeur': {'name': 'FRX/EUR (CAC 40)', 'type': 'index', 'start_year': 2010},
            'ukxgbp': {'name': 'UKX/GBP (FTSE 100)', 'type': 'index', 'start_year': 2010},
            'grxeur': {'name': 'GRX/EUR (DAX 30)', 'type': 'index', 'start_year': 2010},
            'auxaud': {'name': 'AUX/AUD (ASX 200)', 'type': 'index', 'start_year': 2010},
            'hkxhkd': {'name': 'HKX/HKD (Hang Seng)', 'type': 'index', 'start_year': 2010},
            'etxeur': {'name': 'ETX/EUR (EUROSTOXX 50)', 'type': 'index', 'start_year': 2010},
            'udxusd': {'name': 'UDX/USD (US Dollar Index)', 'type': 'index', 'start_year': 2010},
        }
    
    def connect(self) -> bool:
        """建立连接（检查数据目录）"""
        try:
            # 检查数据目录是否存在
            if not self.data_directory.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
            
            # 扫描可用数据文件
            self._build_file_index()
            
            self.connection_status.is_connected = True
            self.connection_status.connected_at = datetime.now()
            self.connection_status.last_error = None
            self.connection_status.retry_count = 0
            
            self.logger.info(f"FXMinuteData connected successfully. Found data for {len(self._file_index)} symbols")
            return True
            
        except Exception as e:
            self.connection_status.is_connected = False
            self.connection_status.last_error = str(e)
            self.connection_status.retry_count += 1
            self.logger.error(f"FXMinuteData connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        self.connection_status.is_connected = False
        self.logger.info("FXMinuteData disconnected")
    
    def _build_file_index(self) -> None:
        """构建文件索引"""
        self._file_index = {}
        
        if not self.data_directory.exists():
            return
        
        # 扫描所有子目录
        for symbol_dir in self.data_directory.iterdir():
            if symbol_dir.is_dir():
                symbol = symbol_dir.name.lower()
                if symbol in self.supported_pairs:
                    self._file_index[symbol] = []
                    
                    # 扫描ZIP文件
                    for zip_file in symbol_dir.glob("DAT_ASCII_*.zip"):
                        # 解析文件名获取年份
                        filename = zip_file.stem
                        parts = filename.split('_')
                        if len(parts) >= 4:
                            try:
                                year = int(parts[-1])
                                self._file_index[symbol].append({
                                    'path': zip_file,
                                    'year': year,
                                    'filename': filename
                                })
                            except ValueError:
                                continue
                    
                    # 按年份排序
                    self._file_index[symbol].sort(key=lambda x: x['year'])
        
        self._index_loaded = True
        self.logger.info(f"Built file index for {len(self._file_index)} symbols")
    
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
            interval: 数据间隔（目前仅支持1分钟）
            
        Returns:
            历史数据DataFrame
        """
        start_time = time.time()
        
        # 仅支持1分钟数据
        if interval != DataInterval.MINUTE_1:
            raise ValueError(f"FXMinuteData only supports 1-minute data, got: {interval}")
        
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        symbol_normalized = self._normalize_symbol(symbol)
        
        try:
            # 检查是否支持该标的
            if symbol_normalized not in self.supported_pairs:
                raise ValueError(f"Symbol '{symbol}' not supported by FXMinuteData")
            
            if not self._index_loaded:
                self._build_file_index()
            
            if symbol_normalized not in self._file_index:
                raise ValueError(f"No data files found for symbol '{symbol}'")
            
            # 获取需要的年份范围
            start_year = start_date.year
            end_year = end_date.year
            
            # 筛选相关的数据文件
            relevant_files = [
                file_info for file_info in self._file_index[symbol_normalized]
                if start_year <= file_info['year'] <= end_year
            ]
            
            if not relevant_files:
                raise ValueError(f"No data files found for {symbol} in date range {start_date.date()} to {end_date.date()}")
            
            # 读取和合并数据
            data_frames = []
            for file_info in relevant_files:
                try:
                    df = self._read_zip_file(file_info['path'], symbol_normalized)
                    if df is not None and not df.empty:
                        # 过滤时间范围 - 使用布尔索引而不是loc切片
                        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
                        if not df_filtered.empty:
                            data_frames.append(df_filtered)
                except Exception as e:
                    self.logger.warning(f"Failed to read {file_info['path']}: {e}")
                    continue
            
            if not data_frames:
                raise ValueError(f"No valid data found for {symbol} in the specified time range")
            
            # 合并所有数据
            combined_data = pd.concat(data_frames, axis=0)
            combined_data.sort_index(inplace=True)
            combined_data.drop_duplicates(inplace=True)
            
            # 标准化DataFrame - 为FeatureEngineer提供大写列名
            combined_data.columns = [col.capitalize() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in combined_data.columns]
            combined_data = DataConverter.clean_data(combined_data)
            
            # 添加数据源信息
            combined_data['source'] = 'fxminute'
            combined_data['symbol'] = symbol_normalized
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            self.logger.info(
                f"Fetched {len(combined_data)} records for {symbol} "
                f"from FXMinuteData in {response_time:.2f}s"
            )
            
            return combined_data
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            self.logger.error(f"Failed to fetch FXMinuteData for {symbol}: {e}")
            raise
    
    def _read_zip_file(self, zip_path: Path, symbol: str) -> Optional[pd.DataFrame]:
        """
        读取ZIP文件中的CSV数据
        
        Args:
            zip_path: ZIP文件路径
            symbol: 标的代码
            
        Returns:
            DataFrame或None
        """
        try:
            # 检查缓存
            cache_file = self.extracted_cache_dir / f"{zip_path.stem}.pkl"
            
            if self.cache_extracted and cache_file.exists():
                # 检查缓存是否比原文件新
                if cache_file.stat().st_mtime > zip_path.stat().st_mtime:
                    try:
                        return pd.read_pickle(cache_file)
                    except Exception as e:
                        self.logger.warning(f"Failed to load cache {cache_file}: {e}")
            
            # 读取ZIP文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 查找CSV文件
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    self.logger.warning(f"No CSV files found in {zip_path}")
                    return None
                
                # 通常只有一个CSV文件
                csv_filename = csv_files[0]
                
                with zip_ref.open(csv_filename) as csv_file:
                    # FX-1-Minute-Data格式: YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME
                    df = pd.read_csv(
                        csv_file,
                        sep=';',
                        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                        dtype={
                            'datetime': str,
                            'open': np.float64,
                            'high': np.float64, 
                            'low': np.float64,
                            'close': np.float64,
                            'volume': np.int64
                        }
                    )
                    
                    # 手动解析日期时间格式：YYYYMMDD HHMMSS
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
                    
                    # 设置时间索引
                    df.set_index('datetime', inplace=True)
                    
                    # 数据验证
                    df = df.dropna()
                    df = df[df['high'] >= df['low']]  # 确保high >= low
                    df = df[df['volume'] >= 0]  # 确保volume >= 0
                    
                    # 缓存提取的数据
                    if self.cache_extracted:
                        try:
                            df.to_pickle(cache_file)
                        except Exception as e:
                            self.logger.warning(f"Failed to cache data to {cache_file}: {e}")
                    
                    return df
        
        except Exception as e:
            self.logger.error(f"Failed to read ZIP file {zip_path}: {e}")
            return None
    
    def fetch_realtime_data(
        self,
        symbols: Union[str, List[str]]
    ) -> Union[MarketData, List[MarketData]]:
        """
        获取实时数据（返回最新历史数据）
        
        Args:
            symbols: 标的代码或代码列表
            
        Returns:
            实时市场数据
        """
        single_symbol = isinstance(symbols, str)
        if single_symbol:
            symbols = [symbols]
        
        results = []
        for symbol in symbols:
            try:
                # 获取最近7天的数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                df = self.fetch_historical_data(
                    symbol, start_date, end_date, DataInterval.MINUTE_1
                )
                
                if df is not None and not df.empty:
                    # 获取最后一条记录
                    last_row = df.iloc[-1]
                    
                    market_data = MarketData(
                        symbol=self._normalize_symbol(symbol),
                        timestamp=df.index[-1].to_pydatetime(),
                        open=last_row['open'],
                        high=last_row['high'],
                        low=last_row['low'],
                        close=last_row['close'],
                        volume=last_row['volume'],
                        source='fxminute',
                        quality=DataQuality.HIGH,  # 高质量历史数据
                        metadata={
                            'is_historical': True,
                            'data_age_hours': (datetime.now() - df.index[-1].to_pydatetime()).total_seconds() / 3600
                        }
                    )
                    results.append(market_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to get latest data for {symbol}: {e}")
                continue
        
        if not results:
            raise ValueError(f"No recent data available for symbols: {symbols}")
        
        return results[0] if single_symbol else results
    
    def validate_symbol(self, symbol: str) -> bool:
        """验证标的代码"""
        normalized_symbol = self._normalize_symbol(symbol)
        return normalized_symbol in self.supported_pairs
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """搜索标的"""
        query_lower = query.lower()
        results = []
        
        for symbol, info in self.supported_pairs.items():
            if (query_lower in symbol or 
                query_lower in info['name'].lower()):
                
                results.append({
                    'symbol': symbol.upper(),
                    'name': info['name'],
                    'type': info['type'],
                    'start_year': str(info['start_year']),
                    'source': 'fxminute'
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_capabilities(self) -> DataSourceCapabilities:
        """获取数据源能力"""
        return DataSourceCapabilities(
            name="FX-1-Minute-Data",
            supported_markets=[
                MarketType.FOREX,
                MarketType.COMMODITIES,
                MarketType.INDEX
            ],
            supported_intervals=[DataInterval.MINUTE_1],
            has_realtime=False,  # 历史数据
            has_historical=True,
            has_streaming=False,
            requires_auth=False,
            is_free=True,
            max_history_days=9000,  # 从2000年开始
            min_interval=DataInterval.MINUTE_1,
            max_symbols_per_request=1,
            rate_limits={},
            data_quality=DataQuality.HIGH,
            latency_ms=None,  # 本地文件读取
            api_version="Local ZIP Files",
            documentation_url="https://github.com/dmidlo/histdata.com-tools",
            support_contact="Local Data Files"
        )
    
    def _normalize_symbol(self, symbol: str) -> str:
        """标准化标的代码"""
        symbol = symbol.lower().strip()
        
        # 移除常见分隔符
        symbol = symbol.replace('/', '').replace('-', '').replace('_', '').replace('.', '')
        
        # 特殊映射
        symbol_mapping = {
            'gold': 'xauusd',
            'silver': 'xagusd',
            'oil': 'wtiusd',
            'brent': 'bcousd',
            'sp500': 'spxusd',
            'nikkei': 'jpxjpy',
            'nasdaq': 'nsxusd',
            'dax': 'grxeur',
            'ftse': 'ukxgbp',
            'cac': 'frxeur',
            'hangseng': 'hkxhkd',
            'asx': 'auxaud',
            'eurostoxx': 'etxeur',
            'dxy': 'udxusd',
        }
        
        return symbol_mapping.get(symbol, symbol)
    
    def get_available_symbols(self) -> List[str]:
        """获取所有可用标的"""
        if not self._index_loaded:
            self._build_file_index()
        return list(self._file_index.keys())
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取标的详细信息"""
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol not in self.supported_pairs:
            return None
        
        info = self.supported_pairs[normalized_symbol].copy()
        
        # 添加可用年份信息
        if normalized_symbol in self._file_index:
            years = [file_info['year'] for file_info in self._file_index[normalized_symbol]]
            info['available_years'] = sorted(years)
            info['data_start'] = min(years) if years else None
            info['data_end'] = max(years) if years else None
        
        return info
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self.connection_status.is_connected:
                return False
            
            # 检查数据目录
            if not self.data_directory.exists():
                return False
            
            # 检查是否有可用数据
            return len(self._file_index) > 0
            
        except Exception as e:
            self.logger.warning(f"FXMinuteData health check failed: {e}")
            return False