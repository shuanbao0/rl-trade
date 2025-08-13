"""
数据格式转换器

提供不同数据格式之间的转换功能，包括：
- MarketData与DataFrame的互转
- Tick数据聚合为OHLC
- 数据标准化和清洗
- 时间序列重采样
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import asdict

from .base import MarketData, DataInterval, DataQuality


class DataConverter:
    """数据格式转换器"""
    
    # 间隔映射表
    INTERVAL_MAP = {
        DataInterval.TICK: None,
        DataInterval.SECOND_1: '1S',
        DataInterval.SECOND_5: '5S',
        DataInterval.SECOND_15: '15S',
        DataInterval.SECOND_30: '30S',
        DataInterval.MINUTE_1: '1T',
        DataInterval.MINUTE_2: '2T',
        DataInterval.MINUTE_5: '5T',
        DataInterval.MINUTE_15: '15T',
        DataInterval.MINUTE_30: '30T',
        DataInterval.MINUTE_90: '90T',
        DataInterval.HOUR_1: '1H',
        DataInterval.HOUR_4: '4H',
        DataInterval.HOUR_6: '6H',
        DataInterval.HOUR_8: '8H',
        DataInterval.HOUR_12: '12H',
        DataInterval.DAY_1: '1D',
        DataInterval.DAY_3: '3D',
        DataInterval.WEEK_1: '1W',
        DataInterval.MONTH_1: '1M',
        DataInterval.MONTH_3: '3M',
        DataInterval.YEAR_1: '1Y'
    }
    
    @staticmethod
    def market_data_to_dict(data: MarketData) -> Dict[str, Any]:
        """
        将MarketData转换为字典
        
        Args:
            data: MarketData实例
            
        Returns:
            字典格式的数据
        """
        return data.to_dict()
    
    @staticmethod
    def dict_to_market_data(data: Dict[str, Any]) -> MarketData:
        """
        将字典转换为MarketData
        
        Args:
            data: 字典格式的数据
            
        Returns:
            MarketData实例
        """
        # 处理质量字段
        if 'quality' in data and isinstance(data['quality'], str):
            data['quality'] = DataQuality(data['quality'])
        elif 'quality' not in data:
            data['quality'] = DataQuality.UNKNOWN
            
        # 处理时间戳
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
        return MarketData(**data)
    
    @staticmethod
    def to_dataframe(
        data: Union[MarketData, List[MarketData]],
        include_metadata: bool = False
    ) -> pd.DataFrame:
        """
        将MarketData转换为DataFrame
        
        Args:
            data: MarketData实例或列表
            include_metadata: 是否包含元数据列
            
        Returns:
            DataFrame格式的数据
        """
        if isinstance(data, MarketData):
            data = [data]
        
        if not data:
            return pd.DataFrame()
        
        # 转换为字典列表
        records = []
        for item in data:
            record = asdict(item)
            record['quality'] = record['quality'].value  # 转换枚举为字符串
            
            if not include_metadata:
                record.pop('metadata', None)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # 设置时间戳为索引
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> List[MarketData]:
        """
        从DataFrame创建MarketData列表
        
        Args:
            df: 数据DataFrame
            symbol: 标的代码（如果DataFrame中没有symbol列）
            
        Returns:
            MarketData列表
        """
        if df.empty:
            return []
        
        # 确保有时间戳索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have datetime index or timestamp column")
        
        results = []
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for timestamp, row in df.iterrows():
            # 检查必需列
            if not all(col in row.index for col in required_columns):
                missing = [col for col in required_columns if col not in row.index]
                raise ValueError(f"Missing required columns: {missing}")
            
            # 构建MarketData
            market_data = MarketData(
                symbol=row.get('symbol', symbol or 'UNKNOWN'),
                timestamp=timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                bid=row.get('bid'),
                ask=row.get('ask'),
                spread=row.get('spread'),
                bid_volume=row.get('bid_volume'),
                ask_volume=row.get('ask_volume'),
                tick_count=row.get('tick_count'),
                vwap=row.get('vwap'),
                turnover=row.get('turnover'),
                source=row.get('source'),
                quality=DataQuality(row.get('quality', 'unknown')),
                metadata=row.get('metadata', {})
            )
            results.append(market_data)
        
        return results
    
    @staticmethod
    def ticks_to_ohlc(
        ticks: List[MarketData],
        interval: DataInterval,
        price_field: str = 'close'
    ) -> pd.DataFrame:
        """
        将Tick数据聚合为OHLC数据
        
        Args:
            ticks: Tick数据列表
            interval: 目标时间间隔
            price_field: 用作价格的字段名
            
        Returns:
            OHLC DataFrame
        """
        if not ticks:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = DataConverter.to_dataframe(ticks)
        
        if interval == DataInterval.TICK:
            return df  # Tick级别不需要聚合
        
        # 获取pandas重采样频率
        freq = DataConverter.INTERVAL_MAP.get(interval)
        if freq is None:
            raise ValueError(f"Unsupported interval for aggregation: {interval}")
        
        # 重采样聚合
        try:
            # 价格数据聚合
            price_col = price_field if price_field in df.columns else 'close'
            
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # 如果只有一个价格字段，构建OHLC
            if price_col != 'close' and 'open' not in df.columns:
                agg_dict = {
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                # 重命名价格列
                df = df.rename(columns={price_col: 'price'})
                df['open'] = df['price']
                df['high'] = df['price']
                df['low'] = df['price']
                df['close'] = df['price']
            
            # 添加其他字段的聚合规则
            if 'bid' in df.columns:
                agg_dict['bid'] = 'last'
            if 'ask' in df.columns:
                agg_dict['ask'] = 'last'
            if 'spread' in df.columns:
                agg_dict['spread'] = 'mean'
            if 'tick_count' in df.columns:
                agg_dict['tick_count'] = 'sum'
            else:
                # 计算每个时间窗口的tick数量
                agg_dict['tick_count'] = 'count'
            
            # 执行重采样
            resampled = df.resample(freq).agg(agg_dict)
            
            # 移除空的时间段
            resampled = resampled.dropna(subset=['close'])
            
            # 计算VWAP
            if 'volume' in resampled.columns and resampled['volume'].sum() > 0:
                # 使用成交量加权平均价
                price_volume = df['close'] * df['volume']
                resampled['vwap'] = price_volume.resample(freq).sum() / resampled['volume']
            else:
                # 使用简单平均价
                resampled['vwap'] = (resampled['high'] + resampled['low'] + resampled['close']) / 3
            
            # 计算价差
            if 'bid' in resampled.columns and 'ask' in resampled.columns:
                resampled['spread'] = resampled['ask'] - resampled['bid']
            
            return resampled
            
        except Exception as e:
            raise ValueError(f"Failed to aggregate ticks to {interval}: {e}")
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化DataFrame列名
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        df = df.copy()
        
        # 列名映射
        column_mapping = {
            # OHLC价格
            'Open': 'open',
            'High': 'high',
            'Low': 'low', 
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            
            # 外汇特有
            'Bid': 'bid',
            'Ask': 'ask',
            'Spread': 'spread',
            'Mid': 'mid_price',
            
            # 时间相关
            'Date': 'date',
            'Time': 'time',
            'Timestamp': 'timestamp',
            'DateTime': 'datetime',
            
            # 其他常见名称
            'Symbol': 'symbol',
            'Instrument': 'symbol',
            'Ticker': 'symbol',
        }
        
        # 应用映射
        df = df.rename(columns=column_mapping)
        
        # 转换为小写（如果还没有映射）
        df.columns = df.columns.str.lower()
        
        return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        df = df.copy()
        
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 前向填充然后后向填充
            df[col] = df[col].ffill().bfill()
        
        # 检查OHLC逻辑
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 修正高低价异常
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # 删除零交易量的异常数据（可选）
        if 'volume' in df.columns:
            # 只删除明显异常的数据，保留零成交量数据（可能是有效的）
            df = df[df['volume'] >= 0]
        
        # 删除价格为0或负数的异常数据
        price_columns = ['open', 'high', 'low', 'close', 'bid', 'ask']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        return df
    
    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        target_interval: DataInterval,
        method: str = 'ohlc'
    ) -> pd.DataFrame:
        """
        重采样数据到目标时间间隔
        
        Args:
            df: 源数据DataFrame
            target_interval: 目标时间间隔
            method: 重采样方法 ('ohlc', 'mean', 'last')
            
        Returns:
            重采样后的DataFrame
        """
        if df.empty:
            return df
        
        freq = DataConverter.INTERVAL_MAP.get(target_interval)
        if freq is None:
            raise ValueError(f"Unsupported target interval: {target_interval}")
        
        if method == 'ohlc':
            # OHLC重采样
            agg_dict = {}
            if 'open' in df.columns:
                agg_dict['open'] = 'first'
            if 'high' in df.columns:
                agg_dict['high'] = 'max'
            if 'low' in df.columns:
                agg_dict['low'] = 'min'
            if 'close' in df.columns:
                agg_dict['close'] = 'last'
            if 'volume' in df.columns:
                agg_dict['volume'] = 'sum'
            
            # 其他数值列使用最后值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in agg_dict:
                    agg_dict[col] = 'last'
            
            result = df.resample(freq).agg(agg_dict)
            
        elif method == 'mean':
            # 平均值重采样
            result = df.resample(freq).mean()
            
        elif method == 'last':
            # 最后值重采样
            result = df.resample(freq).last()
            
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
        
        # 移除空行
        result = result.dropna(how='all')
        
        return result
    
    @staticmethod
    def merge_sources(
        dataframes: Dict[str, pd.DataFrame],
        method: str = 'outer'
    ) -> pd.DataFrame:
        """
        合并多个数据源的数据
        
        Args:
            dataframes: 数据源DataFrame字典 {source_name: df}
            method: 合并方法 ('outer', 'inner', 'average')
            
        Returns:
            合并后的DataFrame
        """
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return list(dataframes.values())[0]
        
        if method in ['outer', 'inner']:
            # 使用pandas的合并功能
            result = None
            for source_name, df in dataframes.items():
                # 添加数据源标识
                df = df.copy()
                df['source'] = source_name
                
                if result is None:
                    result = df
                else:
                    result = pd.merge(
                        result, df,
                        left_index=True, right_index=True,
                        how=method,
                        suffixes=('', f'_{source_name}')
                    )
            
        elif method == 'average':
            # 计算平均值合并
            all_data = []
            for source_name, df in dataframes.items():
                df = df.copy()
                df['source'] = source_name
                all_data.append(df)
            
            combined = pd.concat(all_data)
            
            # 按时间戳分组并计算平均值
            numeric_cols = combined.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_cols}
            agg_dict['source'] = lambda x: ','.join(x.unique())
            
            result = combined.groupby(level=0).agg(agg_dict)
            
        else:
            raise ValueError(f"Unsupported merge method: {method}")
        
        return result
    
    @staticmethod
    def calculate_returns(
        df: pd.DataFrame,
        price_column: str = 'close',
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            df: 价格数据DataFrame
            price_column: 价格列名
            method: 计算方法 ('simple', 'log')
            
        Returns:
            包含收益率的DataFrame
        """
        df = df.copy()
        
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        
        if method == 'simple':
            df['returns'] = df[price_column].pct_change()
        elif method == 'log':
            df['returns'] = np.log(df[price_column] / df[price_column].shift(1))
        else:
            raise ValueError(f"Unsupported return calculation method: {method}")
        
        return df
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        添加基本技术指标
        
        Args:
            df: OHLC数据DataFrame
            
        Returns:
            包含技术指标的DataFrame
        """
        df = df.copy()
        
        if 'close' not in df.columns:
            return df
        
        # 简单移动平均
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # 指数移动平均
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df