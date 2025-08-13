"""
测试TrueFX数据源实现
"""

import pytest
import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests

from src.data.sources.truefx_source import TrueFXDataSource
from src.data.sources.base import DataInterval, MarketData, DataQuality, MarketType


class TestTrueFXDataSource(unittest.TestCase):
    """测试TrueFX数据源"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'name': 'test_truefx',
            'username': None,  # 未认证测试
            'password': None,
            'timeout': 5,
            'stream_interval': 0.5,
            'format': 'csv'
        }
        self.source = TrueFXDataSource(self.config)
        
        # 认证配置
        self.auth_config = {
            'name': 'test_truefx_auth',
            'username': 'test_user',
            'password': 'test_pass',
            'timeout': 5
        }
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.source.name, 'test_truefx')
        self.assertIsNone(self.source.username)
        self.assertIsNone(self.source.password)
        self.assertFalse(self.source.authenticated)
        self.assertIsNone(self.source.session_id)
        self.assertEqual(self.source.timeout, 5)
        self.assertEqual(self.source.stream_interval, 0.5)
    
    def test_currency_pairs_constants(self):
        """测试货币对常量"""
        # 检查未认证用户可用的货币对
        self.assertIn('EUR/USD', TrueFXDataSource.UNAUTHENTICATED_PAIRS)
        self.assertIn('USD/JPY', TrueFXDataSource.UNAUTHENTICATED_PAIRS)
        self.assertEqual(len(TrueFXDataSource.UNAUTHENTICATED_PAIRS), 10)
        
        # 检查认证用户可用的货币对
        self.assertIn('AUD/NZD', TrueFXDataSource.AUTHENTICATED_PAIRS)
        self.assertIn('CAD/CHF', TrueFXDataSource.AUTHENTICATED_PAIRS)
        self.assertGreater(len(TrueFXDataSource.AUTHENTICATED_PAIRS), 
                          len(TrueFXDataSource.UNAUTHENTICATED_PAIRS))
    
    def test_symbol_normalization(self):
        """测试货币对标准化"""
        test_cases = [
            ('EURUSD', 'EUR/USD'),
            ('EUR_USD', 'EUR/USD'),
            ('EUR-USD', 'EUR/USD'),
            ('EUR.USD', 'EUR/USD'),
            ('EUR/USD', 'EUR/USD'),
            ('eurusd', 'EUR/USD'),
            ('GBPJPY', 'GBP/JPY'),
            ('UNKNOWN', 'UNKNOWN')
        ]
        
        for input_symbol, expected in test_cases:
            result = self.source._normalize_symbol(input_symbol)
            self.assertEqual(result, expected, f"Failed for {input_symbol}")
    
    @patch('requests.get')
    def test_unauthenticated_connection(self, mock_get):
        """测试未认证连接"""
        # 模拟成功的未认证连接测试
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "EUR/USD,1609459200000,1.2050,0,1.2053,0,1.2055,1.2048,1.2051"
        mock_get.return_value = mock_response
        
        result = self.source.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.source.connection_status.is_connected)
        self.assertFalse(self.source.authenticated)
        self.assertIsNone(self.source.session_id)
    
    @patch('requests.get')
    def test_authenticated_connection(self, mock_get):
        """测试认证连接"""
        source = TrueFXDataSource(self.auth_config)
        
        # 模拟认证成功
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "test_session_id_12345"
        mock_get.return_value = mock_response
        
        result = source.connect()
        
        self.assertTrue(result)
        self.assertTrue(source.connection_status.is_connected)
        self.assertTrue(source.authenticated)
        self.assertEqual(source.session_id, "test_session_id_12345")
        
        # 验证认证请求参数
        mock_get.assert_called_with(
            f"{TrueFXDataSource.BASE_URL}/connect.html",
            params={
                'u': 'test_user',
                'p': 'test_pass', 
                'q': 'ozrates',
                'f': 'csv'
            },
            timeout=5
        )
    
    @patch('requests.get')
    def test_authentication_failure(self, mock_get):
        """测试认证失败"""
        source = TrueFXDataSource(self.auth_config)
        
        # 模拟认证失败
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "error"
        mock_get.return_value = mock_response
        
        result = source.connect()
        
        self.assertFalse(result)
        self.assertFalse(source.connection_status.is_connected)
        self.assertFalse(source.authenticated)
        self.assertIsNone(source.session_id)
    
    @patch('requests.get')
    def test_disconnect(self, mock_get):
        """测试断开连接"""
        # 设置一个已连接的认证会话
        self.source.session_id = "test_session"
        self.source.authenticated = True
        self.source.connection_status.is_connected = True
        
        # 模拟断开连接请求
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        self.source.disconnect()
        
        self.assertFalse(self.source.connection_status.is_connected)
        self.assertFalse(self.source.authenticated)
        self.assertIsNone(self.source.session_id)
        
        # 验证断开连接请求
        mock_get.assert_called_with(
            f"{TrueFXDataSource.BASE_URL}/disconnect.html",
            params={'di': 'test_session'},
            timeout=5
        )
    
    def test_historical_data_not_supported(self):
        """测试历史数据不支持"""
        with self.assertRaises(NotImplementedError) as context:
            self.source.fetch_historical_data(
                'EUR/USD', '2024-01-01', '2024-01-02', DataInterval.TICK
            )
        
        self.assertIn("manual download", str(context.exception))
        self.assertIn("truefx.com", str(context.exception))
    
    def test_parse_response_success(self):
        """测试解析TrueFX响应数据"""
        # 模拟TrueFX CSV响应
        csv_response = (
            "EUR/USD,1609459200000,1.2050,0,1.2053,0,1.2055,1.2048,1.2051\n"
            "GBP/USD,1609459200500,1.3500,0,1.3503,0,1.3505,1.3498,1.3501"
        )
        
        result = self.source._parse_response(csv_response, ['EUR/USD', 'GBP/USD'])
        
        self.assertEqual(len(result), 2)
        
        # 检查第一个货币对
        eur_usd = result[0]
        self.assertEqual(eur_usd['symbol'], 'EUR/USD')
        self.assertEqual(eur_usd['timestamp'], 1609459200.0)  # 转换为秒
        self.assertEqual(eur_usd['bid'], 1.2050)
        self.assertEqual(eur_usd['ask'], 1.2053)
        self.assertEqual(eur_usd['high'], 1.2055)
        self.assertEqual(eur_usd['low'], 1.2048)
        self.assertEqual(eur_usd['open'], 1.2051)
        
        # 检查第二个货币对
        gbp_usd = result[1]
        self.assertEqual(gbp_usd['symbol'], 'GBP/USD')
        self.assertEqual(gbp_usd['timestamp'], 1609459200.5)
    
    def test_parse_response_malformed(self):
        """测试解析错误格式的响应"""
        # 不完整的CSV行
        malformed_csv = "EUR/USD,1609459200000,1.2050\nGBP/USD"
        
        result = self.source._parse_response(malformed_csv, ['EUR/USD'])
        
        # 应该跳过错误格式的行
        self.assertEqual(len(result), 0)
    
    def test_parse_rate_data(self):
        """测试汇率数据转换为MarketData"""
        rate_data = {
            'symbol': 'EUR/USD',
            'timestamp': 1609459200.0,
            'bid': 1.2050,
            'ask': 1.2053,
            'high': 1.2055,
            'low': 1.2048,
            'open': 1.2051
        }
        
        market_data = self.source._parse_rate_data(rate_data)
        
        self.assertIsInstance(market_data, MarketData)
        self.assertEqual(market_data.symbol, 'EUR/USD')
        self.assertEqual(market_data.timestamp, datetime.fromtimestamp(1609459200.0))
        self.assertEqual(market_data.bid, 1.2050)
        self.assertEqual(market_data.ask, 1.2053)
        self.assertAlmostEqual(market_data.spread, 0.0003, places=4)
        self.assertAlmostEqual(market_data.close, 1.20515, places=5)  # 中间价
        self.assertEqual(market_data.open, 1.2051)
        self.assertEqual(market_data.high, 1.2055)
        self.assertEqual(market_data.low, 1.2048)
        self.assertEqual(market_data.volume, 0)  # TrueFX不提供成交量
        self.assertEqual(market_data.source, 'truefx')
        self.assertEqual(market_data.quality, DataQuality.HIGH)
    
    @patch.object(TrueFXDataSource, '_fetch_rates_data')
    def test_fetch_realtime_data_single(self, mock_fetch):
        """测试单个货币对实时数据获取"""
        # 模拟API响应
        mock_fetch.return_value = [{
            'symbol': 'EUR/USD',
            'timestamp': 1609459200.0,
            'bid': 1.2050,
            'ask': 1.2053,
            'high': 1.2055,
            'low': 1.2048,
            'open': 1.2051
        }]
        
        result = self.source.fetch_realtime_data('EURUSD')
        
        self.assertIsInstance(result, MarketData)
        self.assertEqual(result.symbol, 'EUR/USD')
        mock_fetch.assert_called_once_with(['EUR/USD'])
    
    @patch.object(TrueFXDataSource, '_fetch_rates_data')
    def test_fetch_realtime_data_multiple(self, mock_fetch):
        """测试多个货币对实时数据获取"""
        mock_fetch.return_value = [
            {
                'symbol': 'EUR/USD',
                'timestamp': 1609459200.0,
                'bid': 1.2050,
                'ask': 1.2053,
                'high': 1.2055,
                'low': 1.2048,
                'open': 1.2051
            },
            {
                'symbol': 'GBP/USD',
                'timestamp': 1609459200.5,
                'bid': 1.3500,
                'ask': 1.3503,
                'high': 1.3505,
                'low': 1.3498,
                'open': 1.3501
            }
        ]
        
        result = self.source.fetch_realtime_data(['EUR/USD', 'GBP/USD'])
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].symbol, 'EUR/USD')
        self.assertEqual(result[1].symbol, 'GBP/USD')
    
    def test_validate_symbol_unauthenticated(self):
        """测试未认证用户的货币对验证"""
        # 未认证用户可用的货币对
        self.assertTrue(self.source.validate_symbol('EUR/USD'))
        self.assertTrue(self.source.validate_symbol('EURUSD'))
        self.assertTrue(self.source.validate_symbol('USD/JPY'))
        
        # 仅认证用户可用的货币对
        self.assertFalse(self.source.validate_symbol('AUD/NZD'))
        self.assertFalse(self.source.validate_symbol('CAD/CHF'))
        
        # 无效货币对
        self.assertFalse(self.source.validate_symbol('INVALID/PAIR'))
    
    def test_validate_symbol_authenticated(self):
        """测试认证用户的货币对验证"""
        source = TrueFXDataSource(self.auth_config)
        source.authenticated = True  # 模拟已认证状态
        
        # 基础货币对
        self.assertTrue(source.validate_symbol('EUR/USD'))
        
        # 认证用户专用货币对
        self.assertTrue(source.validate_symbol('AUD/NZD'))
        self.assertTrue(source.validate_symbol('CAD/CHF'))
        
        # 无效货币对
        self.assertFalse(source.validate_symbol('INVALID/PAIR'))
    
    def test_search_symbols(self):
        """测试货币对搜索"""
        # 搜索EUR
        results = self.source.search_symbols('EUR')
        self.assertGreater(len(results), 0)
        self.assertTrue(any('EUR' in r['symbol'] for r in results))
        
        # 搜索USD
        results = self.source.search_symbols('USD', limit=5)
        self.assertLessEqual(len(results), 5)
        self.assertTrue(all(r['type'] == 'forex' for r in results))
        self.assertTrue(all(r['source'] == 'truefx' for r in results))
        
        # 搜索不存在的货币
        results = self.source.search_symbols('XYZ')
        self.assertEqual(len(results), 0)
    
    def test_get_capabilities(self):
        """测试能力获取"""
        caps = self.source.get_capabilities()
        
        self.assertEqual(caps.name, "TrueFX")
        self.assertEqual(caps.supported_markets, [MarketType.FOREX])
        self.assertEqual(caps.supported_intervals, [DataInterval.TICK])
        self.assertTrue(caps.has_realtime)
        self.assertFalse(caps.has_historical)
        self.assertTrue(caps.has_streaming)
        self.assertFalse(caps.requires_auth)
        self.assertTrue(caps.is_free)
        self.assertEqual(caps.min_interval, DataInterval.TICK)
        self.assertEqual(caps.max_symbols_per_request, 15)
        self.assertEqual(caps.data_quality, DataQuality.HIGH)
        self.assertEqual(caps.latency_ms, 500)
    
    @patch('requests.get')
    def test_fetch_rates_data_with_retry(self, mock_get):
        """测试带重试的数据获取"""
        # 第一次请求失败，第二次成功
        mock_get.side_effect = [
            requests.RequestException("Network error"),
            Mock(status_code=200, text="EUR/USD,1609459200000,1.2050,0,1.2053,0,1.2055,1.2048,1.2051")
        ]
        
        # 降低重试延迟加速测试
        self.source.retry_delay = 0.1
        
        result = self.source._fetch_rates_data(['EUR/USD'])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'EUR/USD')
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('requests.get')
    def test_fetch_rates_data_max_retries_exceeded(self, mock_get):
        """测试超过最大重试次数"""
        # 所有请求都失败
        mock_get.side_effect = requests.RequestException("Persistent network error")
        
        # 降低重试次数和延迟加速测试
        self.source.max_retries = 2
        self.source.retry_delay = 0.1
        
        with self.assertRaises(requests.RequestException):
            self.source._fetch_rates_data(['EUR/USD'])
        
        self.assertEqual(mock_get.call_count, 2)
    
    @patch.object(TrueFXDataSource, 'fetch_realtime_data')
    def test_streaming_data(self, mock_fetch):
        """测试流式数据"""
        mock_data = MarketData(
            symbol='EUR/USD',
            timestamp=datetime.now(),
            open=1.2050, high=1.2055, low=1.2048, close=1.2051,
            volume=0, bid=1.2050, ask=1.2053, source='truefx', quality=DataQuality.HIGH
        )
        mock_fetch.return_value = [mock_data]
        
        received_data = []
        
        def callback(data):
            received_data.append(data)
            # 收到数据后停止流
            if len(received_data) >= 2:
                self.source.stop_streaming()
        
        # 启动流式数据（使用很短的间隔）
        self.source.stream_realtime_data(['EUR/USD'], callback, 0.1)
        
        # 等待一些数据
        time.sleep(0.5)
        
        # 验证收到数据
        self.assertGreater(len(received_data), 0)
        for data in received_data:
            if isinstance(data, list):
                self.assertGreater(len(data), 0)
                self.assertIsInstance(data[0], MarketData)
            else:
                self.assertIsInstance(data, MarketData)
    
    @patch.object(TrueFXDataSource, 'fetch_realtime_data')
    def test_health_check(self, mock_fetch):
        """测试健康检查"""
        # 未连接时应该失败
        self.assertFalse(self.source.health_check())
        
        # 连接后但数据获取失败
        self.source.connection_status.is_connected = True
        mock_fetch.side_effect = Exception("Data fetch error")
        self.assertFalse(self.source.health_check())
        
        # 连接正常且数据获取成功
        mock_data = MarketData(
            symbol='EUR/USD', timestamp=datetime.now(),
            open=1.2050, high=1.2055, low=1.2048, close=1.2051,
            volume=0, source='truefx', quality=DataQuality.HIGH
        )
        mock_fetch.return_value = mock_data
        mock_fetch.side_effect = None
        
        self.assertTrue(self.source.health_check())
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(self.source, 'connect', return_value=True) as mock_connect:
            with patch.object(self.source, 'disconnect') as mock_disconnect:
                with self.source as src:
                    self.assertEqual(src, self.source)
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


if __name__ == '__main__':
    unittest.main()