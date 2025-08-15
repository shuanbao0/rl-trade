"""
测试市场处理器功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.market_processors import (
    MarketProcessorFactory,
    StockMarketProcessor,
    ForexMarketProcessor, 
    CryptoMarketProcessor,
    GenericMarketProcessor,
    ProcessingResult
)
from src.data.sources.base import MarketType


class TestProcessingResult(unittest.TestCase):
    """处理结果测试类"""
    
    def test_processing_result_creation(self):
        """测试处理结果创建"""
        data = pd.DataFrame({'Close': [100, 101, 102]})
        warnings = ['Test warning']
        statistics = {'records_processed': 3}
        metadata = {'test': 'value'}
        
        result = ProcessingResult(
            data=data,
            warnings=warnings,
            statistics=statistics,
            metadata=metadata
        )
        
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 3)
        self.assertEqual(result.warnings, warnings)
        self.assertEqual(result.statistics, statistics)
        self.assertEqual(result.metadata, metadata)


class TestMarketProcessorFactory(unittest.TestCase):
    """市场处理器工厂测试类"""
    
    def test_create_stock_processor(self):
        """测试创建股票处理器"""
        processor = MarketProcessorFactory.create_processor(MarketType.STOCK)
        self.assertIsInstance(processor, StockMarketProcessor)
        self.assertEqual(processor.market_type, MarketType.STOCK)
    
    def test_create_forex_processor(self):
        """测试创建外汇处理器"""
        processor = MarketProcessorFactory.create_processor(MarketType.FOREX)
        self.assertIsInstance(processor, ForexMarketProcessor)
        self.assertEqual(processor.market_type, MarketType.FOREX)
    
    def test_create_crypto_processor(self):
        """测试创建加密货币处理器"""
        processor = MarketProcessorFactory.create_processor(MarketType.CRYPTO)
        self.assertIsInstance(processor, CryptoMarketProcessor)
        self.assertEqual(processor.market_type, MarketType.CRYPTO)
    
    def test_create_generic_processor(self):
        """测试创建通用处理器"""
        processor = MarketProcessorFactory.create_processor(MarketType.COMMODITIES)
        self.assertIsInstance(processor, GenericMarketProcessor)
        self.assertEqual(processor.market_type, MarketType.COMMODITIES)
    
    def test_register_custom_processor(self):
        """测试注册自定义处理器"""
        class CustomProcessor(GenericMarketProcessor):
            pass
        
        MarketProcessorFactory.register_processor(MarketType.BONDS, CustomProcessor)
        processor = MarketProcessorFactory.create_processor(MarketType.BONDS)
        self.assertIsInstance(processor, CustomProcessor)
    
    def test_get_supported_markets(self):
        """测试获取支持的市场类型"""
        supported = MarketProcessorFactory.get_supported_markets()
        self.assertIn(MarketType.STOCK, supported)
        self.assertIn(MarketType.FOREX, supported)
        self.assertIn(MarketType.CRYPTO, supported)


class TestStockMarketProcessor(unittest.TestCase):
    """股票市场处理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = StockMarketProcessor()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0],
            'High': [102.0, 103.0, 101.0, 104.0, 100.0, 105.0, 99.0, 106.0, 98.0, 107.0],
            'Low': [99.0, 100.0, 98.0, 101.0, 97.0, 102.0, 96.0, 103.0, 95.0, 104.0],
            'Close': [101.0, 100.0, 102.0, 99.0, 103.0, 98.0, 104.0, 97.0, 105.0, 96.0],
            'Volume': [1000, 1100, 900, 1200, 800, 1300, 700, 1400, 600, 1500],
            'Adj Close': [101.0, 100.0, 102.0, 99.0, 103.0, 98.0, 104.0, 97.0, 105.0, 96.0]
        }, index=dates)
    
    def test_stock_processor_initialization(self):
        """测试股票处理器初始化"""
        self.assertEqual(self.processor.market_type, MarketType.STOCK)
        self.assertIn('apply_split_adjustment', self.processor.config)
        self.assertIn('apply_dividend_adjustment', self.processor.config)
        self.assertIn('price_precision', self.processor.config)
    
    def test_process_valid_stock_data(self):
        """测试处理有效股票数据"""
        result = self.processor.process(self.test_data)
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 10)
        
        # 验证必需列存在
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, result.data.columns)
        
        # 验证元数据
        self.assertEqual(result.metadata['market_type'], 'stock')
        self.assertIn('price_range', result.metadata)
        self.assertIn('volume_range', result.metadata)
    
    def test_process_missing_columns(self):
        """测试处理缺失列的数据"""
        incomplete_data = self.test_data.drop(['Volume'], axis=1)
        result = self.processor.process(incomplete_data)
        
        # 应该有警告
        self.assertTrue(any('Missing required columns' in warning for warning in result.warnings))
        
        # 应该返回原始数据
        self.assertEqual(len(result.data.columns), len(incomplete_data.columns))
    
    def test_process_invalid_ohlc_data(self):
        """测试处理无效OHLC数据"""
        invalid_data = self.test_data.copy()
        # 制造无效的OHLC数据：High < Low
        invalid_data.loc[invalid_data.index[0], 'High'] = 90.0
        invalid_data.loc[invalid_data.index[0], 'Low'] = 95.0
        
        result = self.processor.process(invalid_data)
        
        # 应该移除无效记录
        self.assertLess(len(result.data), len(invalid_data))
        
        # 应该有警告
        self.assertTrue(any('invalid OHLC logic' in warning for warning in result.warnings))
    
    def test_dividend_processing(self):
        """测试分红处理"""
        data_with_dividends = self.test_data.copy()
        data_with_dividends['Dividends'] = [0, 0, 1.0, 0, 0, 0, 0.5, 0, 0, 0]
        
        result = self.processor.process(data_with_dividends)
        
        # 应该识别分红日
        self.assertTrue(any('dividend payment days' in warning for warning in result.warnings))
    
    def test_volume_filtering(self):
        """测试成交量过滤"""
        # 修改配置，设置最小成交量
        self.processor.config['min_volume'] = 1000
        
        data_with_low_volume = self.test_data.copy()
        data_with_low_volume.loc[data_with_low_volume.index[0], 'Volume'] = 500  # 低于阈值
        
        result = self.processor.process(data_with_low_volume)
        
        # 应该移除低成交量记录
        self.assertLess(len(result.data), len(data_with_low_volume))


class TestForexMarketProcessor(unittest.TestCase):
    """外汇市场处理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = ForexMarketProcessor()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        self.test_data = pd.DataFrame({
            'Open': [1.2000, 1.2010, 1.1990, 1.2020, 1.1980, 1.2030, 1.1970, 1.2040, 1.1960, 1.2050],
            'High': [1.2020, 1.2030, 1.2010, 1.2040, 1.2000, 1.2050, 1.1990, 1.2060, 1.1980, 1.2070],
            'Low': [1.1990, 1.2000, 1.1980, 1.2010, 1.1970, 1.2020, 1.1960, 1.2030, 1.1950, 1.2040],
            'Close': [1.2010, 1.1990, 1.2020, 1.1980, 1.2030, 1.1970, 1.2040, 1.1960, 1.2050, 1.1940],
            'Bid': [1.2005, 1.1985, 1.2015, 1.1975, 1.2025, 1.1965, 1.2035, 1.1955, 1.2045, 1.1935],
            'Ask': [1.2015, 1.1995, 1.2025, 1.1985, 1.2035, 1.1975, 1.2045, 1.1965, 1.2055, 1.1945]
        }, index=dates)
    
    def test_forex_processor_initialization(self):
        """测试外汇处理器初始化"""
        self.assertEqual(self.processor.market_type, MarketType.FOREX)
        self.assertIn('calculate_pip_values', self.processor.config)
        self.assertIn('process_spread', self.processor.config)
        self.assertIn('price_precision', self.processor.config)
    
    def test_process_valid_forex_data(self):
        """测试处理有效外汇数据"""
        result = self.processor.process(self.test_data, symbol='EURUSD')
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 10)
        
        # 验证点差计算
        if 'Spread' in result.data.columns:
            self.assertTrue((result.data['Spread'] >= 0).all())
        
        # 验证元数据
        self.assertEqual(result.metadata['market_type'], 'forex')
        self.assertEqual(result.metadata['symbol'], 'EURUSD')
        self.assertIn('precision', result.metadata)
    
    def test_pair_precision_detection(self):
        """测试货币对精度检测"""
        # 测试主要货币对
        eurusd_precision = self.processor._get_pair_precision('EURUSD')
        self.assertEqual(eurusd_precision, 4)
        
        # 测试日元货币对
        usdjpy_precision = self.processor._get_pair_precision('USDJPY')
        self.assertEqual(usdjpy_precision, 2)
        
        # 测试未知货币对
        unknown_precision = self.processor._get_pair_precision('UNKNOWN')
        self.assertEqual(unknown_precision, 5)
    
    def test_spread_processing(self):
        """测试点差处理"""
        result = self.processor.process(self.test_data, symbol='EURUSD')
        
        if 'Spread' in result.data.columns:
            # 验证点差为正数
            self.assertTrue((result.data['Spread'] >= 0).all())
            
            # 验证统计信息
            if 'avg_spread_pips' in result.statistics:
                self.assertGreater(result.statistics['avg_spread_pips'], 0)
    
    def test_invalid_spread_removal(self):
        """测试移除无效点差"""
        invalid_data = self.test_data.copy()
        # 制造无效点差：Ask < Bid
        invalid_data.loc[invalid_data.index[0], 'Ask'] = 1.1990
        invalid_data.loc[invalid_data.index[0], 'Bid'] = 1.2000
        
        result = self.processor.process(invalid_data, symbol='EURUSD')
        
        # 应该移除无效记录
        self.assertLess(len(result.data), len(invalid_data))
        
        # 应该有警告
        self.assertTrue(any('invalid bid/ask spread' in warning for warning in result.warnings))
    
    def test_pip_value_calculation(self):
        """测试点值计算"""
        result = self.processor.process(self.test_data, symbol='EURUSD')
        
        if 'Pip_Value' in result.data.columns:
            # 验证点值
            expected_pip_value = 0.0001  # EURUSD的点值
            self.assertAlmostEqual(result.data['Pip_Value'].iloc[0], expected_pip_value, places=5)


class TestCryptoMarketProcessor(unittest.TestCase):
    """加密货币市场处理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = CryptoMarketProcessor()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        self.test_data = pd.DataFrame({
            'Open': [50000.0, 50100.0, 49900.0, 50200.0, 49800.0, 50300.0, 49700.0, 50400.0, 49600.0, 50500.0],
            'High': [50200.0, 50300.0, 50100.0, 50400.0, 50000.0, 50500.0, 49900.0, 50600.0, 49800.0, 50700.0],
            'Low': [49900.0, 50000.0, 49800.0, 50100.0, 49700.0, 50200.0, 49600.0, 50300.0, 49500.0, 50400.0],
            'Close': [50100.0, 49900.0, 50200.0, 49800.0, 50300.0, 49700.0, 50400.0, 49600.0, 50500.0, 49500.0],
            'Volume': [10.5, 11.2, 9.8, 12.1, 8.9, 13.0, 7.7, 14.2, 6.8, 15.5],
            'Quote_Volume': [525000, 559000, 491000, 603000, 443000, 654000, 382000, 705000, 337000, 767000]
        }, index=dates)
    
    def test_crypto_processor_initialization(self):
        """测试加密货币处理器初始化"""
        self.assertEqual(self.processor.market_type, MarketType.CRYPTO)
        self.assertIn('price_precision', self.processor.config)
        self.assertIn('volume_precision', self.processor.config)
        self.assertIn('remove_flash_crashes', self.processor.config)
    
    def test_process_valid_crypto_data(self):
        """测试处理有效加密货币数据"""
        result = self.processor.process(self.test_data, symbol='BTC-USD')
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 10)
        
        # 验证元数据
        self.assertEqual(result.metadata['market_type'], 'crypto')
        self.assertEqual(result.metadata['symbol'], 'BTC-USD')
        
        # 验证统计信息
        if 'price_volatility' in result.statistics:
            self.assertGreaterEqual(result.statistics['price_volatility'], 0)
    
    def test_flash_crash_detection(self):
        """测试闪崩检测"""
        crash_data = self.test_data.copy()
        # 制造闪崩：价格暴跌60%
        crash_data.loc[crash_data.index[5], 'Close'] = 20000.0
        crash_data.loc[crash_data.index[5], 'Low'] = 20000.0
        
        result = self.processor.process(crash_data, symbol='BTC-USD')
        
        # 应该移除闪崩记录
        self.assertLess(len(result.data), len(crash_data))
        
        # 应该有警告
        self.assertTrue(any('flash crash' in warning for warning in result.warnings))
    
    def test_low_volume_filtering(self):
        """测试低成交量过滤"""
        low_volume_data = self.test_data.copy()
        # 制造极低成交量
        low_volume_data.loc[low_volume_data.index[0], 'Volume'] = 0.0005
        
        result = self.processor.process(low_volume_data, symbol='BTC-USD')
        
        # 应该移除低成交量记录
        self.assertLess(len(result.data), len(low_volume_data))
        
        # 应该有警告
        self.assertTrue(any('very low volume' in warning for warning in result.warnings))
    
    def test_trading_halt_detection(self):
        """测试交易暂停检测"""
        halt_data = self.test_data.copy()
        # 制造交易暂停：成交量为0，价格无变化
        halt_data.loc[halt_data.index[3], 'Volume'] = 0.0
        halt_data.loc[halt_data.index[3], 'Open'] = 50000.0
        halt_data.loc[halt_data.index[3], 'High'] = 50000.0
        halt_data.loc[halt_data.index[3], 'Low'] = 50000.0
        halt_data.loc[halt_data.index[3], 'Close'] = 50000.0
        
        result = self.processor.process(halt_data, symbol='BTC-USD')
        
        # 应该检测到交易暂停
        self.assertTrue(any('trading halt' in warning for warning in result.warnings))
    
    def test_negative_price_handling(self):
        """测试负价格处理"""
        invalid_data = self.test_data.copy()
        # 制造负价格
        invalid_data.loc[invalid_data.index[0], 'Close'] = -100.0
        
        result = self.processor.process(invalid_data, symbol='BTC-USD')
        
        # 应该移除负价格记录
        self.assertLess(len(result.data), len(invalid_data))
        
        # 应该有警告
        self.assertTrue(any('non-positive prices' in warning for warning in result.warnings))


class TestGenericMarketProcessor(unittest.TestCase):
    """通用市场处理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = GenericMarketProcessor(MarketType.COMMODITIES)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0, 102.0, 98.0],
            'High': [102.0, 103.0, 101.0, 104.0, 100.0],
            'Low': [99.0, 100.0, 98.0, 101.0, 97.0],
            'Close': [101.0, 100.0, 102.0, 99.0, 103.0]
        }, index=dates)
    
    def test_generic_processor_initialization(self):
        """测试通用处理器初始化"""
        self.assertEqual(self.processor.market_type, MarketType.COMMODITIES)
        self.assertIn('price_precision', self.processor.config)
        self.assertIn('remove_outliers', self.processor.config)
        self.assertIn('fill_missing', self.processor.config)
    
    def test_process_generic_data(self):
        """测试处理通用数据"""
        result = self.processor.process(self.test_data)
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertEqual(len(result.data), 5)
        
        # 验证元数据
        self.assertEqual(result.metadata['market_type'], 'commodities')
    
    def test_outlier_removal(self):
        """测试异常值移除"""
        outlier_data = self.test_data.copy()
        # 添加异常值
        outlier_data.loc[outlier_data.index[0], 'Close'] = 1000.0  # 极大异常值
        
        result = self.processor.process(outlier_data)
        
        # 异常值应该被移除（如果启用了移除功能）
        if self.processor.config['remove_outliers']:
            max_close = result.data['Close'].max()
            self.assertLess(max_close, 500.0)  # 应该小于异常值


if __name__ == '__main__':
    unittest.main()