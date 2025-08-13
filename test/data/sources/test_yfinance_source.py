"""
测试YFinance数据源实现
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.sources.yfinance_source import YFinanceDataSource
from src.data.sources.base import DataInterval, MarketData, DataQuality


class TestYFinanceDataSource(unittest.TestCase):
    """测试YFinance数据源"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'name': 'test_yfinance',
            'proxy': None,
            'auto_adjust_interval': True,
            'poll_interval': 1.0
        }
        self.source = YFinanceDataSource(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.source.name, 'test_yfinance')
        self.assertTrue(self.source.auto_adjust_interval)
        self.assertEqual(self.source.poll_interval, 1.0)
        self.assertIsNotNone(self.source.data_manager)
    
    def test_proxy_setup(self):
        """测试代理设置"""
        config_with_proxy = {
            'proxy': 'socks5://127.0.0.1:7891'
        }
        
        with patch('src.data.sources.yfinance_source.yf') as mock_yf:
            mock_yf.set_config = Mock()
            source = YFinanceDataSource(config_with_proxy)
            mock_yf.set_config.assert_called_once_with(proxy='socks5://127.0.0.1:7891')
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_connection(self, mock_ticker):
        """测试连接功能"""
        # 模拟成功连接
        mock_instance = Mock()
        mock_instance.info = {'symbol': 'AAPL', 'shortName': 'Apple Inc.'}
        mock_ticker.return_value = mock_instance
        
        result = self.source.connect()
        self.assertTrue(result)
        self.assertTrue(self.source.connection_status.is_connected)
        
        # 测试断开连接
        self.source.disconnect()
        self.assertFalse(self.source.connection_status.is_connected)
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_connection_failure(self, mock_ticker):
        """测试连接失败"""
        # 模拟连接失败
        mock_instance = Mock()
        mock_instance.info = {}  # 空信息表示失败
        mock_ticker.return_value = mock_instance
        
        result = self.source.connect()
        self.assertFalse(result)
        self.assertFalse(self.source.connection_status.is_connected)
        self.assertIsNotNone(self.source.connection_status.last_error)
    
    def test_interval_conversion(self):
        """测试时间间隔转换"""
        # 测试支持的间隔
        test_cases = [
            (DataInterval.MINUTE_1, "1m"),
            (DataInterval.MINUTE_5, "5m"),
            (DataInterval.HOUR_1, "1h"),
            (DataInterval.DAY_1, "1d"),
            (DataInterval.WEEK_1, "1wk"),
            (DataInterval.MONTH_1, "1mo")
        ]
        
        for interval, expected in test_cases:
            result = self.source._convert_interval_to_yfinance(interval)
            self.assertEqual(result, expected)
        
        # 测试不支持的间隔
        with self.assertRaises(ValueError):
            self.source._convert_interval_to_yfinance(DataInterval.TICK)
    
    def test_period_conversion(self):
        """测试期间转换"""
        test_cases = [
            (5, "7d"),
            (20, "1mo"),
            (70, "3mo"),
            (150, "6mo"),
            (300, "1y"),
            (600, "2y"),
            (1500, "5y"),
            (3000, "10y"),
            (5000, "max")
        ]
        
        for days, expected in test_cases:
            result = self.source._days_to_period_string(days)
            self.assertEqual(result, expected)
    
    def test_symbol_normalization(self):
        """测试标的代码标准化"""
        test_cases = [
            ('aapl', 'AAPL'),
            ('AAPL', 'AAPL'),
            ('eurusd', 'EURUSD=X'),
            ('EURUSD', 'EURUSD=X'),
            ('btc', 'BTC-USD'),
            ('BTC', 'BTC-USD'),
            ('GOOGL', 'GOOGL')  # 已经是正确格式
        ]
        
        for input_symbol, expected in test_cases:
            result = self.source._normalize_symbol(input_symbol)
            self.assertEqual(result, expected)
    
    @patch('src.data.sources.yfinance_source.DataManager')
    def test_fetch_historical_data(self, mock_data_manager_class):
        """测试历史数据获取"""
        # 准备模拟数据
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        # 设置mock实例
        mock_data_manager_instance = Mock()
        mock_data_manager_instance.get_stock_data = Mock(return_value=mock_data)
        mock_data_manager_class.return_value = mock_data_manager_instance
        
        # 重新创建source以使用mock的DataManager
        source = YFinanceDataSource(self.config)
        
        # 执行测试
        result = source.fetch_historical_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-03',
            interval=DataInterval.DAY_1
        )
        
        # 验证结果
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn('open', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('source', result.columns)
        self.assertEqual(result['source'].iloc[0], 'yfinance')
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_fetch_realtime_data_single(self, mock_ticker):
        """测试单个标的实时数据获取"""
        # 准备模拟数据
        mock_info = {
            'symbol': 'AAPL',
            'regularMarketPrice': 150.0,
            'regularMarketOpen': 148.0,
            'regularMarketDayHigh': 152.0,
            'regularMarketDayLow': 147.0,
            'regularMarketVolume': 1000000,
            'bid': 149.8,
            'ask': 150.2,
            'currency': 'USD',
            'exchange': 'NASDAQ'
        }
        
        mock_instance = Mock()
        mock_instance.info = mock_info
        mock_ticker.return_value = mock_instance
        
        # 执行测试
        result = self.source.fetch_realtime_data('AAPL')
        
        # 验证结果
        self.assertIsInstance(result, MarketData)
        self.assertEqual(result.symbol, 'AAPL')
        self.assertEqual(result.close, 150.0)
        self.assertEqual(result.open, 148.0)
        self.assertEqual(result.high, 152.0)
        self.assertEqual(result.low, 147.0)
        self.assertEqual(result.volume, 1000000)
        self.assertEqual(result.bid, 149.8)
        self.assertEqual(result.ask, 150.2)
        self.assertAlmostEqual(result.spread, 0.4, places=1)
        self.assertEqual(result.source, 'yfinance')
        self.assertEqual(result.quality, DataQuality.MEDIUM)
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_fetch_realtime_data_multiple(self, mock_ticker):
        """测试多个标的实时数据获取"""
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'regularMarketPrice': 100.0,
                'regularMarketOpen': 99.0,
                'regularMarketDayHigh': 101.0,
                'regularMarketDayLow': 98.0,
                'regularMarketVolume': 1000
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # 执行测试
        result = self.source.fetch_realtime_data(['AAPL', 'GOOGL'])
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], MarketData)
        self.assertIsInstance(result[1], MarketData)
        self.assertEqual(result[0].symbol, 'AAPL')
        self.assertEqual(result[1].symbol, 'GOOGL')
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_validate_symbol(self, mock_ticker):
        """测试标的验证"""
        # 有效标的
        mock_instance = Mock()
        mock_instance.info = {'symbol': 'AAPL'}
        mock_ticker.return_value = mock_instance
        
        result = self.source.validate_symbol('AAPL')
        self.assertTrue(result)
        
        # 无效标的
        mock_instance.info = {}
        result = self.source.validate_symbol('INVALID')
        self.assertFalse(result)
    
    def test_search_symbols(self):
        """测试标的搜索"""
        # 搜索Apple
        results = self.source.search_symbols('AAPL')
        self.assertIsInstance(results, list)
        self.assertTrue(any('AAPL' in r['symbol'] for r in results))
        
        # 搜索外汇
        results = self.source.search_symbols('EUR')
        self.assertTrue(any('EUR' in r['symbol'] for r in results))
        
        # 搜索Bitcoin
        results = self.source.search_symbols('BTC')
        self.assertTrue(any('BTC' in r['symbol'] for r in results))
    
    def test_get_capabilities(self):
        """测试能力获取"""
        caps = self.source.get_capabilities()
        
        self.assertEqual(caps.name, "YFinance")
        self.assertTrue(caps.has_realtime)
        self.assertTrue(caps.has_historical)
        self.assertTrue(caps.has_streaming)
        self.assertFalse(caps.requires_auth)
        self.assertTrue(caps.is_free)
        
        # 检查支持的市场
        self.assertIn('stock', [m.value for m in caps.supported_markets])
        self.assertIn('forex', [m.value for m in caps.supported_markets])
        
        # 检查支持的间隔
        supported_intervals = [i.value for i in caps.supported_intervals]
        self.assertIn('1m', supported_intervals)
        self.assertIn('1h', supported_intervals)
        self.assertIn('1d', supported_intervals)
    
    @patch('src.data.sources.yfinance_source.yf.Ticker')
    def test_streaming_data(self, mock_ticker):
        """测试流式数据"""
        # 准备模拟数据
        mock_instance = Mock()
        mock_instance.info = {
            'symbol': 'AAPL',
            'regularMarketPrice': 150.0,
            'regularMarketOpen': 148.0,
            'regularMarketDayHigh': 152.0,
            'regularMarketDayLow': 147.0,
            'regularMarketVolume': 1000000
        }
        mock_ticker.return_value = mock_instance
        
        # 收集流式数据
        received_data = []
        
        def callback(data):
            received_data.append(data)
            # 收到几个数据点后停止
            if len(received_data) >= 2:
                self.source.stop_streaming()
        
        # 启动流式数据（使用很短的间隔）
        self.source.stream_realtime_data(['AAPL'], callback, 0.1)
        
        # 等待数据
        import time
        time.sleep(0.5)
        
        # 验证收到数据
        self.assertGreater(len(received_data), 0)
        for data in received_data:
            # 流式数据回调可能返回单个MarketData或列表
            if isinstance(data, list):
                self.assertGreater(len(data), 0)
                self.assertIsInstance(data[0], MarketData)
                self.assertEqual(data[0].symbol, 'AAPL')
            else:
                self.assertIsInstance(data, MarketData)
                self.assertEqual(data.symbol, 'AAPL')
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(self.source, 'connect', return_value=True) as mock_connect:
            with patch.object(self.source, 'disconnect') as mock_disconnect:
                with self.source as src:
                    self.assertEqual(src, self.source)
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    def test_statistics_tracking(self):
        """测试统计跟踪"""
        initial_stats = self.source.get_statistics()
        self.assertEqual(initial_stats['requests_total'], 0)
        
        with patch('src.data.sources.yfinance_source.yf.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.info = {
                'regularMarketPrice': 100,
                'regularMarketOpen': 99,
                'regularMarketDayHigh': 101,
                'regularMarketDayLow': 98,
                'regularMarketVolume': 1000
            }
            mock_ticker.return_value = mock_instance
            
            # 执行请求
            self.source.fetch_realtime_data('AAPL')
            
            # 检查统计更新
            updated_stats = self.source.get_statistics()
            self.assertGreater(updated_stats['requests_total'], 0)
            self.assertGreater(updated_stats['requests_success'], 0)
            # 允许响应时间为0（模拟情况下可能很快）
            self.assertGreaterEqual(updated_stats['avg_response_time'], 0)


if __name__ == '__main__':
    unittest.main()