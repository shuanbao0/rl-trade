"""
市场特定数据处理器

针对不同市场类型提供专门的数据标准化和清洗逻辑：
1. 股票市场：复权处理、分红拆股处理
2. 外汇市场：点差处理、买卖价处理、点值计算
3. 加密货币：价格精度处理、交易量标准化
4. 商品市场：结算价处理、持仓量处理
5. 通用处理：数据验证、异常值处理
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..sources.base import MarketType, DataInterval, MarketData
from ...utils.logger import setup_logger


@dataclass
class ProcessingResult:
    """数据处理结果"""
    data: pd.DataFrame                  # 处理后的数据
    warnings: List[str]                 # 处理警告
    statistics: Dict[str, Any]          # 处理统计信息
    metadata: Dict[str, Any]            # 元数据信息


class BaseMarketProcessor(ABC):
    """市场数据处理器基类"""
    
    def __init__(self, market_type: MarketType):
        """
        初始化处理器
        
        Args:
            market_type: 市场类型
        """
        self.market_type = market_type
        self.logger = setup_logger(f"MarketProcessor.{market_type.value}")
        
        # 处理配置
        self.config = self._get_default_config()
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        pass
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessingResult:
        """
        处理市场数据
        
        Args:
            data: 原始数据
            **kwargs: 额外参数
            
        Returns:
            处理结果
        """
        pass
    
    def _validate_required_columns(self, data: pd.DataFrame, required_columns: List[str]) -> List[str]:
        """验证必需的列是否存在"""
        missing_columns = []
        for col in required_columns:
            if col not in data.columns:
                missing_columns.append(col)
        return missing_columns
    
    def _remove_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """移除异常值"""
        cleaned_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                if outliers.any():
                    self.logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                    cleaned_data = cleaned_data[~outliers]
            
            elif method == 'zscore':
                z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                outliers = z_scores > 3
                if outliers.any():
                    self.logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                    cleaned_data = cleaned_data[~outliers]
        
        return cleaned_data
    
    def _fill_missing_values(self, data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """填充缺失值"""
        filled_data = data.copy()
        
        if method == 'forward':
            filled_data = filled_data.ffill()
        elif method == 'backward':
            filled_data = filled_data.bfill()
        elif method == 'interpolate':
            numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
            filled_data[numeric_columns] = filled_data[numeric_columns].interpolate()
        
        return filled_data
    
    def _calculate_statistics(self, original_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """计算处理统计信息"""
        return {
            'original_records': len(original_data),
            'processed_records': len(processed_data),
            'records_removed': len(original_data) - len(processed_data),
            'removal_rate': (len(original_data) - len(processed_data)) / len(original_data) if len(original_data) > 0 else 0,
            'processing_timestamp': datetime.now().isoformat()
        }


class StockMarketProcessor(BaseMarketProcessor):
    """股票市场数据处理器"""
    
    def __init__(self):
        super().__init__(MarketType.STOCK)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取股票市场默认配置"""
        return {
            'apply_split_adjustment': True,      # 是否应用拆股调整
            'apply_dividend_adjustment': True,   # 是否应用分红调整
            'remove_price_outliers': False,      # 是否移除价格异常值
            'min_volume': 0,                     # 最小成交量
            'price_precision': 2,                # 价格精度
            'volume_precision': 0                # 成交量精度
        }
    
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessingResult:
        """处理股票市场数据"""
        self.logger.debug(f"Processing stock market data: {len(data)} records")
        
        warnings = []
        processed_data = data.copy()
        
        # 验证必需列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = self._validate_required_columns(data, required_columns)
        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")
            return ProcessingResult(data, warnings, {}, {})
        
        # 1. 价格数据标准化
        processed_data = self._standardize_prices(processed_data)
        
        # 2. 成交量处理
        processed_data = self._process_volume(processed_data)
        
        # 3. 复权处理
        if self.config['apply_split_adjustment'] and 'Adj Close' in data.columns:
            processed_data = self._apply_adjustment(processed_data, warnings)
        
        # 4. 分红处理
        if self.config['apply_dividend_adjustment'] and 'Dividends' in data.columns:
            processed_data = self._process_dividends(processed_data, warnings)
        
        # 5. 移除异常值
        if self.config['remove_price_outliers']:
            original_len = len(processed_data)
            processed_data = self._remove_outliers(
                processed_data, 
                ['Open', 'High', 'Low', 'Close'], 
                method='iqr'
            )
            if len(processed_data) < original_len:
                warnings.append(f"Removed {original_len - len(processed_data)} price outliers")
        
        # 6. 数据验证
        processed_data = self._validate_stock_data(processed_data, warnings)
        
        # 7. 填充缺失值
        processed_data = self._fill_missing_values(processed_data, method='forward')
        
        # 计算统计信息
        statistics = self._calculate_statistics(data, processed_data)
        
        # 元数据
        metadata = {
            'market_type': self.market_type.value,
            'processor_config': self.config,
            'price_range': {
                'min': float(processed_data[['Open', 'High', 'Low', 'Close']].min().min()),
                'max': float(processed_data[['Open', 'High', 'Low', 'Close']].max().max())
            },
            'volume_range': {
                'min': float(processed_data['Volume'].min()),
                'max': float(processed_data['Volume'].max())
            }
        }
        
        self.logger.debug(f"Stock processing completed: {len(processed_data)} records")
        
        return ProcessingResult(processed_data, warnings, statistics, metadata)
    
    def _standardize_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化价格数据"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        precision = self.config['price_precision']
        
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].round(precision)
        
        return data
    
    def _process_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理成交量数据"""
        if 'Volume' in data.columns:
            # 移除负成交量
            data = data[data['Volume'] >= 0]
            
            # 应用最小成交量过滤
            min_volume = self.config['min_volume']
            if min_volume > 0:
                data = data[data['Volume'] >= min_volume]
            
            # 标准化成交量精度
            precision = self.config['volume_precision']
            data['Volume'] = data['Volume'].round(precision)
        
        return data
    
    def _apply_adjustment(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """应用复权调整"""
        if 'Adj Close' not in data.columns:
            warnings.append("Adj Close column not found, skipping adjustment")
            return data
        
        # 计算调整因子
        data['Adj_Factor'] = data['Adj Close'] / data['Close']
        
        # 应用调整因子到OHLC价格
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data[f'Adj_{col}'] = data[col] * data['Adj_Factor']
        
        return data
    
    def _process_dividends(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """处理分红数据"""
        if 'Dividends' in data.columns:
            dividend_days = data[data['Dividends'] > 0]
            if not dividend_days.empty:
                warnings.append(f"Found {len(dividend_days)} dividend payment days")
                # 可以在这里添加分红除权处理逻辑
        
        return data
    
    def _validate_stock_data(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """验证股票数据的逻辑正确性"""
        validated_data = data.copy()
        
        # 验证OHLC逻辑
        invalid_ohlc = (
            (validated_data['High'] < validated_data['Low']) |
            (validated_data['High'] < validated_data['Open']) |
            (validated_data['High'] < validated_data['Close']) |
            (validated_data['Low'] > validated_data['Open']) |
            (validated_data['Low'] > validated_data['Close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            warnings.append(f"Found {invalid_count} records with invalid OHLC logic")
            validated_data = validated_data[~invalid_ohlc]
        
        return validated_data


class ForexMarketProcessor(BaseMarketProcessor):
    """外汇市场数据处理器"""
    
    def __init__(self):
        super().__init__(MarketType.FOREX)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取外汇市场默认配置"""
        return {
            'calculate_pip_values': True,        # 是否计算点值
            'process_spread': True,              # 是否处理点差
            'price_precision': 5,                # 价格精度（通常5位小数）
            'major_pairs_precision': 4,          # 主要货币对精度
            'jpy_pairs_precision': 2,            # 日元货币对精度
            'remove_weekend_gaps': True,         # 是否移除周末跳空
            'spread_threshold': 50,              # 点差阈值（点）
            'min_pip_movement': 0.00001          # 最小点移动
        }
    
    def process(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> ProcessingResult:
        """处理外汇市场数据"""
        self.logger.debug(f"Processing forex market data: {len(data)} records for {symbol}")
        
        warnings = []
        processed_data = data.copy()
        
        # 验证必需列
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = self._validate_required_columns(data, required_columns)
        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")
            return ProcessingResult(data, warnings, {}, {})
        
        # 1. 确定货币对精度
        precision = self._get_pair_precision(symbol)
        
        # 2. 价格标准化
        processed_data = self._standardize_forex_prices(processed_data, precision)
        
        # 3. 处理买卖价差
        if self.config['process_spread'] and 'Bid' in data.columns and 'Ask' in data.columns:
            processed_data = self._process_bid_ask_spread(processed_data, warnings)
        
        # 4. 计算点值
        if self.config['calculate_pip_values']:
            processed_data = self._calculate_pip_values(processed_data, symbol)
        
        # 5. 移除周末跳空
        if self.config['remove_weekend_gaps']:
            processed_data = self._remove_weekend_gaps(processed_data, warnings)
        
        # 6. 外汇特定验证
        processed_data = self._validate_forex_data(processed_data, warnings)
        
        # 7. 填充缺失值
        processed_data = self._fill_missing_values(processed_data, method='interpolate')
        
        # 计算统计信息
        statistics = self._calculate_statistics(data, processed_data)
        
        # 添加外汇特定统计
        if 'Spread' in processed_data.columns:
            statistics['avg_spread_pips'] = float(processed_data['Spread'].mean() * (10 ** precision))
            statistics['max_spread_pips'] = float(processed_data['Spread'].max() * (10 ** precision))
        
        # 元数据
        metadata = {
            'market_type': self.market_type.value,
            'symbol': symbol,
            'precision': precision,
            'processor_config': self.config,
            'price_range': {
                'min': float(processed_data[['Open', 'High', 'Low', 'Close']].min().min()),
                'max': float(processed_data[['Open', 'High', 'Low', 'Close']].max().max())
            }
        }
        
        self.logger.debug(f"Forex processing completed: {len(processed_data)} records")
        
        return ProcessingResult(processed_data, warnings, statistics, metadata)
    
    def _get_pair_precision(self, symbol: str) -> int:
        """根据货币对确定价格精度"""
        if not symbol:
            return self.config['price_precision']
        
        symbol = symbol.upper()
        
        # 日元货币对通常是2位小数
        if 'JPY' in symbol:
            return self.config['jpy_pairs_precision']
        
        # 主要货币对通常是4位小数
        major_pairs = ['EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        if symbol in major_pairs:
            return self.config['major_pairs_precision']
        
        # 默认精度
        return self.config['price_precision']
    
    def _standardize_forex_prices(self, data: pd.DataFrame, precision: int) -> pd.DataFrame:
        """标准化外汇价格"""
        price_columns = ['Open', 'High', 'Low', 'Close', 'Bid', 'Ask']
        
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].round(precision)
        
        return data
    
    def _process_bid_ask_spread(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """处理买卖价差"""
        if 'Bid' in data.columns and 'Ask' in data.columns:
            # 计算点差
            data['Spread'] = data['Ask'] - data['Bid']
            
            # 检查异常点差
            spread_threshold = self.config['spread_threshold'] / 10000  # 转换为价格单位
            large_spreads = data['Spread'] > spread_threshold
            
            if large_spreads.any():
                warnings.append(f"Found {large_spreads.sum()} records with large spreads (>{self.config['spread_threshold']} pips)")
                # 可以选择移除或调整异常点差
                data = data[~large_spreads]
        
        return data
    
    def _calculate_pip_values(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """计算点值"""
        if not symbol:
            return data
        
        precision = self._get_pair_precision(symbol)
        pip_size = 10 ** (-precision)
        
        if 'Close' in data.columns:
            # 计算点值（简化版本）
            data['Pip_Value'] = pip_size
            
            # 如果有价格变动，计算点数变化
            if len(data) > 1:
                data['Price_Change_Pips'] = data['Close'].diff() / pip_size
        
        return data
    
    def _remove_weekend_gaps(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """移除周末跳空"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # 识别周末跳空（周五收盘到周一开盘之间的大幅价格变动）
        # 这是一个简化的实现
        data['Weekday'] = data.index.weekday
        
        # 找到周一的数据点
        monday_data = data[data['Weekday'] == 0]
        
        if not monday_data.empty:
            # 可以在这里添加跳空检测和处理逻辑
            pass
        
        return data
    
    def _validate_forex_data(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """验证外汇数据"""
        validated_data = data.copy()
        
        # 验证OHLC逻辑
        invalid_ohlc = (
            (validated_data['High'] < validated_data['Low']) |
            (validated_data['High'] < validated_data['Open']) |
            (validated_data['High'] < validated_data['Close']) |
            (validated_data['Low'] > validated_data['Open']) |
            (validated_data['Low'] > validated_data['Close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            warnings.append(f"Found {invalid_count} records with invalid OHLC logic")
            validated_data = validated_data[~invalid_ohlc]
        
        # 验证买卖价逻辑
        if 'Bid' in validated_data.columns and 'Ask' in validated_data.columns:
            invalid_spread = validated_data['Ask'] <= validated_data['Bid']
            if invalid_spread.any():
                invalid_count = invalid_spread.sum()
                warnings.append(f"Found {invalid_count} records with invalid bid/ask spread")
                validated_data = validated_data[~invalid_spread]
        
        return validated_data


class CryptoMarketProcessor(BaseMarketProcessor):
    """加密货币市场数据处理器"""
    
    def __init__(self):
        super().__init__(MarketType.CRYPTO)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取加密货币市场默认配置"""
        return {
            'price_precision': 8,                # 价格精度（最多8位小数）
            'volume_precision': 6,               # 成交量精度
            'remove_flash_crashes': False,       # 是否移除闪崩数据
            'max_price_change_percent': 50,      # 最大价格变动百分比
            'min_volume_threshold': 0.0001,      # 最小成交量阈值
            'handle_trading_halts': True,        # 是否处理交易暂停
            'normalize_volume_currency': True    # 是否标准化成交量货币
        }
    
    def process(self, data: pd.DataFrame, symbol: str = None, **kwargs) -> ProcessingResult:
        """处理加密货币市场数据"""
        self.logger.debug(f"Processing crypto market data: {len(data)} records for {symbol}")
        
        warnings = []
        processed_data = data.copy()
        
        # 验证必需列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = self._validate_required_columns(data, required_columns)
        if missing_columns:
            warnings.append(f"Missing required columns: {missing_columns}")
            return ProcessingResult(data, warnings, {}, {})
        
        # 1. 价格精度标准化
        processed_data = self._standardize_crypto_prices(processed_data)
        
        # 2. 成交量处理
        processed_data = self._process_crypto_volume(processed_data, warnings)
        
        # 3. 移除闪崩数据
        if self.config['remove_flash_crashes']:
            processed_data = self._remove_flash_crashes(processed_data, warnings)
        
        # 4. 处理交易暂停
        if self.config['handle_trading_halts']:
            processed_data = self._handle_trading_halts(processed_data, warnings)
        
        # 5. 加密货币特定验证
        processed_data = self._validate_crypto_data(processed_data, warnings)
        
        # 6. 填充缺失值
        processed_data = self._fill_missing_values(processed_data, method='interpolate')
        
        # 计算统计信息
        statistics = self._calculate_statistics(data, processed_data)
        
        # 添加加密货币特定统计
        if len(processed_data) > 1:
            price_volatility = processed_data['Close'].pct_change().std()
            statistics['price_volatility'] = float(price_volatility)
            statistics['volume_volatility'] = float(processed_data['Volume'].pct_change().std())
        
        # 元数据
        metadata = {
            'market_type': self.market_type.value,
            'symbol': symbol,
            'processor_config': self.config,
            'price_range': {
                'min': float(processed_data[['Open', 'High', 'Low', 'Close']].min().min()),
                'max': float(processed_data[['Open', 'High', 'Low', 'Close']].max().max())
            },
            'volume_range': {
                'min': float(processed_data['Volume'].min()),
                'max': float(processed_data['Volume'].max())
            }
        }
        
        self.logger.debug(f"Crypto processing completed: {len(processed_data)} records")
        
        return ProcessingResult(processed_data, warnings, statistics, metadata)
    
    def _standardize_crypto_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化加密货币价格"""
        price_columns = ['Open', 'High', 'Low', 'Close']
        precision = self.config['price_precision']
        
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].round(precision)
        
        return data
    
    def _process_crypto_volume(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """处理加密货币成交量"""
        if 'Volume' in data.columns:
            # 移除极小成交量
            min_volume = self.config['min_volume_threshold']
            low_volume = data['Volume'] < min_volume
            
            if low_volume.any():
                warnings.append(f"Found {low_volume.sum()} records with very low volume")
                data = data[~low_volume].copy()
            
            # 标准化成交量精度
            precision = self.config['volume_precision']
            data['Volume'] = data['Volume'].round(precision)
            
            # 处理报价货币成交量（如果存在）
            if 'Quote_Volume' in data.columns:
                data['Quote_Volume'] = data['Quote_Volume'].round(precision)
        
        return data
    
    def _remove_flash_crashes(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """移除闪崩数据"""
        if len(data) < 2:
            return data
        
        # 计算价格变动百分比
        price_change_pct = data['Close'].pct_change().abs() * 100
        max_change = self.config['max_price_change_percent']
        
        # 识别异常变动
        flash_crashes = price_change_pct > max_change
        
        if flash_crashes.any():
            crash_count = flash_crashes.sum()
            warnings.append(f"Removed {crash_count} potential flash crash records")
            data = data[~flash_crashes]
        
        return data
    
    def _handle_trading_halts(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """处理交易暂停"""
        # 识别可能的交易暂停（成交量为0且价格无变化）
        if 'Volume' in data.columns and len(data) > 1:
            zero_volume = data['Volume'] == 0
            price_unchanged = (
                (data['Open'] == data['Close']) & 
                (data['High'] == data['Low']) & 
                (data['Open'] == data['High'])
            )
            
            trading_halts = zero_volume & price_unchanged
            
            if trading_halts.any():
                halt_count = trading_halts.sum()
                warnings.append(f"Found {halt_count} potential trading halt periods")
                # 可以选择保留或移除这些数据点
        
        return data
    
    def _validate_crypto_data(self, data: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
        """验证加密货币数据"""
        validated_data = data.copy()
        
        # 验证OHLC逻辑
        invalid_ohlc = (
            (validated_data['High'] < validated_data['Low']) |
            (validated_data['High'] < validated_data['Open']) |
            (validated_data['High'] < validated_data['Close']) |
            (validated_data['Low'] > validated_data['Open']) |
            (validated_data['Low'] > validated_data['Close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            warnings.append(f"Found {invalid_count} records with invalid OHLC logic")
            validated_data = validated_data[~invalid_ohlc]
        
        # 验证价格为正数
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in validated_data.columns:
                negative_prices = validated_data[col] <= 0
                if negative_prices.any():
                    warnings.append(f"Found {negative_prices.sum()} non-positive prices in {col}")
                    validated_data = validated_data[~negative_prices]
        
        return validated_data


class MarketProcessorFactory:
    """市场处理器工厂"""
    
    _processors = {
        MarketType.STOCK: StockMarketProcessor,
        MarketType.FOREX: ForexMarketProcessor,
        MarketType.CRYPTO: CryptoMarketProcessor,
        # 可以为其他市场类型添加处理器
    }
    
    @classmethod
    def create_processor(cls, market_type: MarketType) -> BaseMarketProcessor:
        """
        创建市场处理器
        
        Args:
            market_type: 市场类型
            
        Returns:
            市场处理器实例
        """
        processor_class = cls._processors.get(market_type)
        
        if processor_class is None:
            # 如果没有专门的处理器，返回通用处理器
            return GenericMarketProcessor(market_type)
        
        # 检查是否是通用处理器或其子类
        if issubclass(processor_class, GenericMarketProcessor):
            return processor_class(market_type)
        else:
            return processor_class()
    
    @classmethod
    def register_processor(cls, market_type: MarketType, processor_class: type):
        """
        注册自定义处理器
        
        Args:
            market_type: 市场类型
            processor_class: 处理器类
        """
        cls._processors[market_type] = processor_class
    
    @classmethod
    def get_supported_markets(cls) -> List[MarketType]:
        """获取支持的市场类型"""
        return list(cls._processors.keys())


class GenericMarketProcessor(BaseMarketProcessor):
    """通用市场处理器（用于没有专门处理器的市场类型）"""
    
    def __init__(self, market_type: MarketType):
        super().__init__(market_type)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取通用默认配置"""
        return {
            'price_precision': 4,
            'volume_precision': 2,
            'remove_outliers': True,
            'fill_missing': True
        }
    
    def process(self, data: pd.DataFrame, **kwargs) -> ProcessingResult:
        """通用数据处理"""
        self.logger.debug(f"Processing {self.market_type.value} market data: {len(data)} records")
        
        warnings = []
        processed_data = data.copy()
        
        # 基本数据清洗
        if self.config['remove_outliers']:
            price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in data.columns]
            if price_columns:
                processed_data = self._remove_outliers(processed_data, price_columns)
        
        if self.config['fill_missing']:
            processed_data = self._fill_missing_values(processed_data)
        
        # 计算统计信息
        statistics = self._calculate_statistics(data, processed_data)
        
        metadata = {
            'market_type': self.market_type.value,
            'processor_config': self.config
        }
        
        return ProcessingResult(processed_data, warnings, statistics, metadata)