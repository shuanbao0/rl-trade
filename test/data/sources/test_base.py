"""
测试数据源基础类和数据结构
"""

import pytest
import unittest
import time
from datetime import datetime
import pandas as pd
import numpy as np

from src.data.sources.base import (
    DataInterval, MarketType, MarketData, DataSourceCapabilities,
    DataQuality, AbstractDataSource, ConnectionStatus
)


class TestDataInterval(unittest.TestCase):
    """测试数据间隔枚举"""
    
    def test_interval_values(self):
        """测试间隔值"""
        self.assertEqual(DataInterval.TICK.value, "tick")
        self.assertEqual(DataInterval.MINUTE_1.value, "1m")
        self.assertEqual(DataInterval.HOUR_1.value, "1h")
        self.assertEqual(DataInterval.DAY_1.value, "1d")
    
    def test_interval_comparison(self):
        """测试间隔比较"""
        self.assertEqual(DataInterval.MINUTE_5, DataInterval.MINUTE_5)
        self.assertNotEqual(DataInterval.MINUTE_1, DataInterval.MINUTE_5)


class TestMarketType(unittest.TestCase):
    """测试市场类型枚举"""
    
    def test_market_values(self):
        """测试市场类型值"""
        self.assertEqual(MarketType.STOCK.value, "stock")
        self.assertEqual(MarketType.FOREX.value, "forex")
        self.assertEqual(MarketType.CRYPTO.value, "crypto")


class TestMarketData(unittest.TestCase):
    """测试MarketData数据类"""
    
    def setUp(self):
        """设置测试数据"""
        self.sample_data = MarketData(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
    
    def test_market_data_creation(self):
        """测试MarketData创建"""
        data = self.sample_data
        self.assertEqual(data.symbol, "AAPL")
        self.assertEqual(data.open, 150.0)
        self.assertEqual(data.high, 155.0)
        self.assertEqual(data.low, 149.0)
        self.assertEqual(data.close, 154.0)
        self.assertEqual(data.volume, 1000000)
    
    def test_spread_calculation(self):
        """测试价差自动计算"""
        data = MarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=0,
            bid=1.1003,
            ask=1.1007
        )
        self.assertAlmostEqual(data.spread, 0.0004, places=4)
    
    def test_ohlc_validation(self):
        """测试OHLC逻辑验证"""
        with self.assertRaises(ValueError):
            # High价格低于Close
            MarketData(
                symbol="TEST",
                timestamp=datetime.now(),
                open=100.0,
                high=99.0,  # 错误：高价低于开盘价
                low=98.0,
                close=101.0,
                volume=1000
            )
        
        with self.assertRaises(ValueError):
            # Low价格高于Close
            MarketData(
                symbol="TEST",
                timestamp=datetime.now(),
                open=100.0,
                high=102.0,
                low=101.0,  # 错误：低价高于开盘价
                close=99.0,
                volume=1000
            )
    
    def test_vwap_calculation(self):
        """测试VWAP计算"""
        data = MarketData(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000
        )
        # VWAP应该是(高+低+收)/3
        expected_vwap = (105.0 + 98.0 + 103.0) / 3
        self.assertAlmostEqual(data.vwap, expected_vwap, places=2)
    
    def test_to_dict(self):
        """测试转换为字典"""
        data_dict = self.sample_data.to_dict()
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict['symbol'], "AAPL")
        self.assertEqual(data_dict['open'], 150.0)
        self.assertIn('timestamp', data_dict)
        self.assertIn('quality', data_dict)


class TestDataSourceCapabilities(unittest.TestCase):
    """测试数据源能力描述"""
    
    def setUp(self):
        """设置测试数据"""
        self.capabilities = DataSourceCapabilities(
            name="TestSource",
            supported_markets=[MarketType.STOCK, MarketType.FOREX],
            supported_intervals=[DataInterval.MINUTE_1, DataInterval.HOUR_1],
            has_realtime=True,
            has_historical=True,
            rate_limits={"requests_per_minute": 100}
        )
    
    def test_capabilities_creation(self):
        """测试能力描述创建"""
        caps = self.capabilities
        self.assertEqual(caps.name, "TestSource")
        self.assertTrue(caps.has_realtime)
        self.assertTrue(caps.has_historical)
        self.assertEqual(len(caps.supported_markets), 2)
        self.assertEqual(len(caps.supported_intervals), 2)
    
    def test_supports_market(self):
        """测试市场支持检查"""
        caps = self.capabilities
        self.assertTrue(caps.supports_market(MarketType.STOCK))
        self.assertTrue(caps.supports_market(MarketType.FOREX))
        self.assertFalse(caps.supports_market(MarketType.CRYPTO))
    
    def test_supports_interval(self):
        """测试间隔支持检查"""
        caps = self.capabilities
        self.assertTrue(caps.supports_interval(DataInterval.MINUTE_1))
        self.assertTrue(caps.supports_interval(DataInterval.HOUR_1))
        self.assertFalse(caps.supports_interval(DataInterval.DAY_1))
    
    def test_get_rate_limit(self):
        """测试速率限制获取"""
        caps = self.capabilities
        self.assertEqual(caps.get_rate_limit("requests_per_minute"), 100)
        self.assertIsNone(caps.get_rate_limit("unknown_endpoint"))


class TestConnectionStatus(unittest.TestCase):
    """测试连接状态"""
    
    def test_connection_status_creation(self):
        """测试连接状态创建"""
        status = ConnectionStatus()
        self.assertFalse(status.is_connected)
        self.assertIsNone(status.connected_at)
        self.assertIsNone(status.last_error)
        self.assertEqual(status.retry_count, 0)
        self.assertEqual(status.health_score, 1.0)
    
    def test_connection_status_update(self):
        """测试连接状态更新"""
        status = ConnectionStatus()
        status.is_connected = True
        status.connected_at = datetime.now()
        status.health_score = 0.8
        
        self.assertTrue(status.is_connected)
        self.assertIsNotNone(status.connected_at)
        self.assertEqual(status.health_score, 0.8)


class MockDataSource(AbstractDataSource):
    """模拟数据源用于测试"""
    
    def connect(self) -> bool:
        self.connection_status.is_connected = True
        return True
    
    def disconnect(self) -> None:
        self.connection_status.is_connected = False
    
    def fetch_historical_data(self, symbol, start_date, end_date, interval):
        # 返回模拟数据
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 102,
            'low': np.random.randn(len(dates)) + 98,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        return data
    
    def fetch_realtime_data(self, symbols):
        start_time = time.time()
        
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            results = []
            for symbol in symbols:
                data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=100.0,
                    high=102.0,
                    low=98.0,
                    close=101.0,
                    volume=5000
                )
                results.append(data)
            
            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(True, response_time)
            
            return results[0] if len(results) == 1 else results
        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time)
            raise
    
    def validate_symbol(self, symbol):
        return symbol.upper() in ['AAPL', 'GOOGL', 'MSFT', 'EURUSD']
    
    def get_capabilities(self):
        return DataSourceCapabilities(
            name="MockSource",
            supported_markets=[MarketType.STOCK],
            supported_intervals=[DataInterval.MINUTE_1, DataInterval.HOUR_1],
            has_realtime=True,
            has_historical=True
        )


class TestAbstractDataSource(unittest.TestCase):
    """测试抽象数据源基类"""
    
    def setUp(self):
        """设置测试"""
        self.source = MockDataSource({
            'name': 'test_source',
            'test_param': 'test_value'
        })
    
    def test_source_creation(self):
        """测试数据源创建"""
        self.assertEqual(self.source.name, 'test_source')
        self.assertEqual(self.source.config['test_param'], 'test_value')
        self.assertIsNotNone(self.source.logger)
    
    def test_connection_management(self):
        """测试连接管理"""
        # 初始状态
        self.assertFalse(self.source.connection_status.is_connected)
        
        # 连接
        result = self.source.connect()
        self.assertTrue(result)
        self.assertTrue(self.source.connection_status.is_connected)
        
        # 断开
        self.source.disconnect()
        self.assertFalse(self.source.connection_status.is_connected)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with self.source as src:
            self.assertTrue(src.connection_status.is_connected)
        self.assertFalse(self.source.connection_status.is_connected)
    
    def test_historical_data_fetch(self):
        """测试历史数据获取"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        data = self.source.fetch_historical_data(
            'AAPL', start_date, end_date, DataInterval.HOUR_1
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertIn('open', data.columns)
        self.assertIn('close', data.columns)
    
    def test_realtime_data_fetch(self):
        """测试实时数据获取"""
        # 单个标的
        data = self.source.fetch_realtime_data('AAPL')
        self.assertIsInstance(data, MarketData)
        self.assertEqual(data.symbol, 'AAPL')
        
        # 多个标的
        data_list = self.source.fetch_realtime_data(['AAPL', 'GOOGL'])
        self.assertIsInstance(data_list, list)
        self.assertEqual(len(data_list), 2)
        self.assertIsInstance(data_list[0], MarketData)
    
    def test_symbol_validation(self):
        """测试标的验证"""
        self.assertTrue(self.source.validate_symbol('AAPL'))
        self.assertTrue(self.source.validate_symbol('aapl'))  # 大小写不敏感
        self.assertFalse(self.source.validate_symbol('INVALID'))
    
    def test_capabilities(self):
        """测试能力获取"""
        caps = self.source.get_capabilities()
        self.assertIsInstance(caps, DataSourceCapabilities)
        self.assertEqual(caps.name, "MockSource")
        self.assertTrue(caps.has_realtime)
        self.assertTrue(caps.has_historical)
    
    def test_statistics(self):
        """测试统计信息"""
        initial_stats = self.source.get_statistics()
        self.assertEqual(initial_stats['requests_total'], 0)
        
        # 执行一些请求
        self.source.fetch_realtime_data('AAPL')
        
        updated_stats = self.source.get_statistics()
        self.assertGreater(updated_stats['requests_total'], 0)
    
    def test_health_check(self):
        """测试健康检查"""
        # 未连接时应该失败
        self.assertFalse(self.source.health_check())
        
        # 连接后应该成功
        self.source.connect()
        self.assertTrue(self.source.health_check())
    
    def test_streaming_simulation(self):
        """测试流式数据模拟"""
        received_data = []
        
        def callback(data):
            received_data.append(data)
        
        # 启动流式数据（使用很短的间隔进行测试）
        self.source.stream_realtime_data(['AAPL'], callback, 0.1)
        
        # 等待一些数据
        import time
        time.sleep(0.5)
        
        # 停止流式数据
        self.source.stop_streaming()
        
        # 验证收到了数据
        self.assertGreater(len(received_data), 0)
        for data in received_data:
            self.assertIsInstance(data, MarketData)


if __name__ == '__main__':
    unittest.main()