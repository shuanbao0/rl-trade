"""
测试Oanda数据源实现
"""

import pytest
import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import requests
import pandas as pd

from src.data.sources.oanda_source import OandaDataSource
from src.data.sources.base import DataInterval, MarketData, DataQuality, MarketType


class TestOandaDataSource(unittest.TestCase):
    """测试Oanda数据源"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'name': 'test_oanda',
            'access_token': 'test_access_token_123',
            'account_id': 'test_account_123',
            'environment': 'practice',
            'timeout': 10,
            'rate_limit': 120
        }
        self.source = OandaDataSource(self.config)
        
        # 生产环境配置
        self.live_config = {
            'name': 'test_oanda_live',
            'access_token': 'live_token_456',
            'account_id': 'live_account_456',
            'environment': 'live',
            'timeout': 15
        }
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.source.name, 'test_oanda')
        self.assertEqual(self.source.access_token, 'test_access_token_123')
        self.assertEqual(self.source.account_id, 'test_account_123')
        self.assertEqual(self.source.environment, 'practice')
        self.assertEqual(self.source.api_url, OandaDataSource.PRACTICE_API_URL)
        self.assertEqual(self.source.stream_url, OandaDataSource.PRACTICE_STREAM_URL)
        
        # 测试认证头
        self.assertIn('Authorization', self.source.headers)
        self.assertIn('Bearer test_access_token_123', self.source.headers['Authorization'])
    
    def test_initialization_live_environment(self):
        """测试生产环境初始化"""
        live_source = OandaDataSource(self.live_config)
        
        self.assertEqual(live_source.environment, 'live')
        self.assertEqual(live_source.api_url, OandaDataSource.LIVE_API_URL)
        self.assertEqual(live_source.stream_url, OandaDataSource.LIVE_STREAM_URL)
    
    def test_initialization_missing_credentials(self):
        """测试缺失认证信息"""
        # 缺少access_token
        with self.assertRaises(ValueError) as context:
            OandaDataSource({'account_id': 'test'})
        self.assertIn('access_token is required', str(context.exception))
        
        # 缺少account_id
        with self.assertRaises(ValueError) as context:
            OandaDataSource({'access_token': 'test'})
        self.assertIn('account_id is required', str(context.exception))
    
    def test_initialization_invalid_environment(self):
        """测试无效环境"""
        invalid_config = self.config.copy()
        invalid_config['environment'] = 'invalid'
        
        with self.assertRaises(ValueError) as context:
            OandaDataSource(invalid_config)
        self.assertIn("environment must be 'practice' or 'live'", str(context.exception))
    
    @patch('requests.get')
    def test_successful_connection(self, mock_get):
        """测试成功连接"""
        # 模拟账户信息响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'account': {
                'currency': 'USD',
                'balance': '10000.0000'
            }
        }
        mock_get.return_value = mock_response
        
        result = self.source.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.source.connection_status.is_connected)
        self.assertIsNotNone(self.source.connection_status.connected_at)
        self.assertIsNone(self.source.connection_status.last_error)
        
        # 验证请求参数
        mock_get.assert_called_with(
            f"{self.source.api_url}/v3/accounts/{self.source.account_id}",
            headers=self.source.headers,
            timeout=self.source.timeout
        )
    
    @patch('requests.get')
    def test_connection_failure(self, mock_get):
        """测试连接失败"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response
        
        result = self.source.connect()
        
        self.assertFalse(result)
        self.assertFalse(self.source.connection_status.is_connected)
        self.assertIsNotNone(self.source.connection_status.last_error)
        self.assertGreater(self.source.connection_status.retry_count, 0)
    
    def test_disconnect(self):
        """测试断开连接"""
        # 设置连接状态
        self.source.connection_status.is_connected = True
        self.source._streaming = True
        
        with patch.object(self.source, 'stop_streaming') as mock_stop:
            self.source.disconnect()
            
            mock_stop.assert_called_once()
            self.assertFalse(self.source.connection_status.is_connected)
    
    def test_symbol_normalization(self):
        """测试货币对标准化"""
        test_cases = [
            ('EURUSD', 'EUR_USD'),
            ('EUR/USD', 'EUR_USD'),
            ('EUR-USD', 'EUR_USD'),
            ('EUR.USD', 'EUR_USD'),
            ('EUR_USD', 'EUR_USD'),
            ('eurusd', 'EUR_USD'),
            ('GBPJPY', 'GBP_JPY'),
            ('XAU_USD', 'XAU_USD'),  # 黄金
            ('US30_USD', 'US30_USD')  # 道琼斯指数
        ]
        
        for input_symbol, expected in test_cases:
            result = self.source._normalize_symbol(input_symbol)
            self.assertEqual(result, expected, f"Failed for {input_symbol}")
    
    def test_interval_mapping(self):
        """测试时间间隔映射"""
        test_cases = [
            (DataInterval.SECOND_5, 'S5'),
            (DataInterval.MINUTE_1, 'M1'),
            (DataInterval.MINUTE_15, 'M15'),
            (DataInterval.HOUR_1, 'H1'),
            (DataInterval.DAY_1, 'D'),
            (DataInterval.WEEK_1, 'W'),
            (DataInterval.MONTH_1, 'M')
        ]
        
        for interval, expected in test_cases:
            result = self.source.INTERVAL_MAP.get(interval)
            self.assertEqual(result, expected)
    
    @patch.object(OandaDataSource, '_fetch_candles_data')
    def test_fetch_historical_data(self, mock_fetch_candles):
        """测试历史数据获取"""
        # 模拟蜡烛图数据响应
        mock_candles = [
            {
                'time': '2024-01-01T10:00:00.000000Z',
                'complete': True,
                'volume': 100,
                'mid': {
                    'o': '1.2050',
                    'h': '1.2055',
                    'l': '1.2048',
                    'c': '1.2052'
                },
                'bid': {'c': '1.2051'},
                'ask': {'c': '1.2053'}
            },
            {
                'time': '2024-01-01T11:00:00.000000Z',
                'complete': True,
                'volume': 150,
                'mid': {
                    'o': '1.2052',
                    'h': '1.2058',
                    'l': '1.2050',
                    'c': '1.2056'
                },
                'bid': {'c': '1.2055'},
                'ask': {'c': '1.2057'}
            }
        ]
        mock_fetch_candles.return_value = mock_candles
        
        # 测试数据获取
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 12, 0, 0)
        
        result = self.source.fetch_historical_data(
            'EUR/USD', start_date, end_date, DataInterval.HOUR_1
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
        self.assertIn('bid', result.columns)
        self.assertIn('ask', result.columns)
        self.assertIn('spread', result.columns)
        
        # 验证数值
        self.assertEqual(result.iloc[0]['open'], 1.2050)
        self.assertEqual(result.iloc[0]['close'], 1.2052)
        self.assertEqual(result.iloc[0]['bid'], 1.2051)
        self.assertEqual(result.iloc[0]['ask'], 1.2053)
        self.assertAlmostEqual(result.iloc[0]['spread'], 0.0002, places=4)
    
    def test_fetch_historical_data_unsupported_interval(self):
        """测试不支持的时间间隔"""
        with self.assertRaises(ValueError) as context:
            self.source.fetch_historical_data(
                'EUR/USD', '2024-01-01', '2024-01-02', DataInterval.TICK
            )
        self.assertIn('Unsupported interval', str(context.exception))
    
    @patch.object(OandaDataSource, '_fetch_pricing_data')
    def test_fetch_realtime_data_single(self, mock_fetch_pricing):
        """测试单个货币对实时数据获取"""
        mock_pricing_data = [{
            'instrument': 'EUR_USD',
            'time': '2024-01-01T12:00:00.000000Z',
            'bids': [{'price': '1.2050', 'liquidity': 1000000}],
            'asks': [{'price': '1.2053', 'liquidity': 1000000}],
            'tradeable': True
        }]
        mock_fetch_pricing.return_value = mock_pricing_data
        
        result = self.source.fetch_realtime_data('EURUSD')
        
        self.assertIsInstance(result, MarketData)
        self.assertEqual(result.symbol, 'EUR_USD')
        self.assertEqual(result.bid, 1.2050)
        self.assertEqual(result.ask, 1.2053)
        self.assertAlmostEqual(result.spread, 0.0003, places=4)
        self.assertAlmostEqual(result.close, 1.20515, places=5)  # 中间价
        self.assertEqual(result.bid_volume, 1000000)
        self.assertEqual(result.ask_volume, 1000000)
        self.assertEqual(result.source, 'oanda')
        self.assertEqual(result.quality, DataQuality.HIGH)
    
    @patch.object(OandaDataSource, '_fetch_pricing_data')
    def test_fetch_realtime_data_multiple(self, mock_fetch_pricing):
        """测试多个货币对实时数据获取"""
        mock_pricing_data = [
            {
                'instrument': 'EUR_USD',
                'time': '2024-01-01T12:00:00.000000Z',
                'bids': [{'price': '1.2050'}],
                'asks': [{'price': '1.2053'}]
            },
            {
                'instrument': 'GBP_USD',
                'time': '2024-01-01T12:00:00.000000Z',
                'bids': [{'price': '1.3500'}],
                'asks': [{'price': '1.3503'}]
            }
        ]
        mock_fetch_pricing.return_value = mock_pricing_data
        
        result = self.source.fetch_realtime_data(['EUR/USD', 'GBP/USD'])
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].symbol, 'EUR_USD')
        self.assertEqual(result[1].symbol, 'GBP_USD')
    
    @patch('requests.get')
    def test_fetch_candles_data_with_retry(self, mock_get):
        """测试带重试的蜡烛图数据获取"""
        # 第一次请求失败，第二次成功
        mock_responses = [
            Mock(status_code=429),  # 速率限制
            Mock(status_code=200, json=lambda: {'candles': [{'test': 'data'}]})
        ]
        mock_get.side_effect = mock_responses
        
        # 降低重试延迟加速测试
        self.source.retry_delay = 0.1
        
        result = self.source._fetch_candles_data('EUR_USD', {'granularity': 'H1'})
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['test'], 'data')
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('requests.get')
    def test_fetch_pricing_data_max_retries_exceeded(self, mock_get):
        """测试超过最大重试次数"""
        # 所有请求都返回速率限制错误
        mock_get.return_value = Mock(status_code=429)
        
        # 降低重试次数和延迟加速测试
        self.source.max_retries = 2
        self.source.retry_delay = 0.1
        
        with self.assertRaises(requests.RequestException):
            self.source._fetch_pricing_data(['EUR_USD'])
        
        self.assertEqual(mock_get.call_count, 2)
    
    def test_parse_candles_to_dataframe(self):
        """测试蜡烛图数据解析"""
        candles = [
            {
                'time': '2024-01-01T10:00:00.000000Z',
                'complete': True,
                'volume': 100,
                'mid': {
                    'o': '1.2050',
                    'h': '1.2055',
                    'l': '1.2048',
                    'c': '1.2052'
                },
                'bid': {'c': '1.2051'},
                'ask': {'c': '1.2053'}
            },
            {
                'time': '2024-01-01T11:00:00.000000Z',
                'complete': False,  # 未完成的蜡烛图应该被跳过
                'volume': 50,
                'mid': {
                    'o': '1.2052',
                    'h': '1.2054',
                    'l': '1.2051',
                    'c': '1.2053'
                }
            }
        ]
        
        df = self.source._parse_candles_to_dataframe(candles, 'EUR_USD')
        
        # 只有完成的蜡烛图被包含
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['open'], 1.2050)
        self.assertEqual(df.iloc[0]['high'], 1.2055)
        self.assertEqual(df.iloc[0]['low'], 1.2048)
        self.assertEqual(df.iloc[0]['close'], 1.2052)
        self.assertEqual(df.iloc[0]['volume'], 100)
        self.assertEqual(df.iloc[0]['bid'], 1.2051)
        self.assertEqual(df.iloc[0]['ask'], 1.2053)
        self.assertAlmostEqual(df.iloc[0]['spread'], 0.0002, places=4)
        self.assertEqual(df.iloc[0]['symbol'], 'EUR_USD')
        self.assertEqual(df.iloc[0]['source'], 'oanda')
    
    def test_parse_pricing_data(self):
        """测试价格数据解析"""
        price_info = {
            'instrument': 'EUR_USD',
            'time': '2024-01-01T12:00:00.000000Z',
            'bids': [{'price': '1.2050', 'liquidity': 1000000}],
            'asks': [{'price': '1.2053', 'liquidity': 1500000}],
            'tradeable': True,
            'unitsAvailable': {
                'default': {'long': '100000', 'short': '100000'}
            }
        }
        
        market_data = self.source._parse_pricing_data(price_info)
        
        self.assertIsInstance(market_data, MarketData)
        self.assertEqual(market_data.symbol, 'EUR_USD')
        self.assertEqual(market_data.bid, 1.2050)
        self.assertEqual(market_data.ask, 1.2053)
        self.assertAlmostEqual(market_data.close, 1.20515, places=5)  # 中间价
        self.assertAlmostEqual(market_data.spread, 0.0003, places=4)
        self.assertEqual(market_data.bid_volume, 1000000)
        self.assertEqual(market_data.ask_volume, 1500000)
        self.assertEqual(market_data.source, 'oanda')
        self.assertEqual(market_data.quality, DataQuality.HIGH)
        
        # 检查元数据
        self.assertEqual(market_data.metadata['account_id'], self.source.account_id)
        self.assertEqual(market_data.metadata['environment'], self.source.environment)
        self.assertTrue(market_data.metadata['tradeable'])
    
    def test_parse_pricing_data_invalid(self):
        """测试无效价格数据解析"""
        # 没有买卖价的数据
        invalid_price_info = {
            'instrument': 'EUR_USD',
            'time': '2024-01-01T12:00:00.000000Z'
            # 缺少bids和asks
        }
        
        result = self.source._parse_pricing_data(invalid_price_info)
        self.assertIsNone(result)
    
    @patch.object(OandaDataSource, '_get_instruments')
    def test_validate_symbol(self, mock_get_instruments):
        """测试货币对验证"""
        mock_get_instruments.return_value = {
            'EUR_USD': {'displayName': 'EUR/USD'},
            'GBP_USD': {'displayName': 'GBP/USD'},
            'USD_JPY': {'displayName': 'USD/JPY'}
        }
        
        self.assertTrue(self.source.validate_symbol('EUR/USD'))
        self.assertTrue(self.source.validate_symbol('EURUSD'))
        self.assertTrue(self.source.validate_symbol('eur_usd'))
        self.assertFalse(self.source.validate_symbol('INVALID_PAIR'))
    
    @patch.object(OandaDataSource, '_get_instruments')
    def test_search_symbols(self, mock_get_instruments):
        """测试货币对搜索"""
        mock_get_instruments.return_value = {
            'EUR_USD': {'displayName': 'EUR/USD', 'type': 'CURRENCY'},
            'EUR_GBP': {'displayName': 'EUR/GBP', 'type': 'CURRENCY'},
            'EUR_JPY': {'displayName': 'EUR/JPY', 'type': 'CURRENCY'},
            'GBP_USD': {'displayName': 'GBP/USD', 'type': 'CURRENCY'},
            'XAU_USD': {'displayName': 'Gold', 'type': 'METAL'}
        }
        
        # 搜索EUR
        results = self.source.search_symbols('EUR', limit=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all('EUR' in r['symbol'] for r in results))
        self.assertTrue(all(r['source'] == 'oanda' for r in results))
        
        # 搜索Gold
        results = self.source.search_symbols('Gold')
        self.assertGreaterEqual(len(results), 1)  # 可能找到多个包含Gold的结果
        found_gold = any(r['name'] == 'Gold' for r in results)
        self.assertTrue(found_gold)
    
    def test_get_capabilities(self):
        """测试能力获取"""
        caps = self.source.get_capabilities()
        
        self.assertEqual(caps.name, "OANDA")
        self.assertEqual(set(caps.supported_markets), {MarketType.FOREX, MarketType.COMMODITIES})
        self.assertIn(DataInterval.MINUTE_1, caps.supported_intervals)
        self.assertIn(DataInterval.HOUR_1, caps.supported_intervals)
        self.assertIn(DataInterval.DAY_1, caps.supported_intervals)
        self.assertTrue(caps.has_realtime)
        self.assertTrue(caps.has_historical)
        self.assertTrue(caps.has_streaming)
        self.assertTrue(caps.requires_auth)
        self.assertFalse(caps.is_free)
        self.assertEqual(caps.min_interval, DataInterval.SECOND_5)
        self.assertEqual(caps.max_symbols_per_request, 20)
        self.assertEqual(caps.data_quality, DataQuality.HIGH)
        self.assertEqual(caps.latency_ms, 50)
        self.assertEqual(caps.rate_limits['requests_per_second'], 120)
    
    @patch('requests.get')
    def test_get_instruments_with_caching(self, mock_get):
        """测试产品列表获取和缓存"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'instruments': [
                {
                    'name': 'EUR_USD',
                    'displayName': 'EUR/USD',
                    'type': 'CURRENCY',
                    'marginRate': '0.0333',
                    'pipLocation': -4,
                    'displayPrecision': 5
                },
                {
                    'name': 'XAU_USD',
                    'displayName': 'Gold',
                    'type': 'METAL',
                    'marginRate': '0.05',
                    'pipLocation': -2,
                    'displayPrecision': 3
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # 第一次调用
        instruments1 = self.source._get_instruments()
        self.assertEqual(len(instruments1), 2)
        self.assertIn('EUR_USD', instruments1)
        self.assertIn('XAU_USD', instruments1)
        self.assertEqual(instruments1['EUR_USD']['displayName'], 'EUR/USD')
        self.assertEqual(instruments1['XAU_USD']['type'], 'METAL')
        
        # 第二次调用应该使用缓存
        instruments2 = self.source._get_instruments()
        self.assertEqual(instruments1, instruments2)
        
        # 只应该调用一次API
        self.assertEqual(mock_get.call_count, 1)
    
    @patch('requests.get')
    def test_health_check_success(self, mock_get):
        """测试健康检查成功"""
        self.source.connection_status.is_connected = True
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.source.health_check()
        self.assertTrue(result)
    
    @patch('requests.get')
    def test_health_check_failure(self, mock_get):
        """测试健康检查失败"""
        self.source.connection_status.is_connected = True
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = self.source.health_check()
        self.assertFalse(result)
    
    def test_health_check_not_connected(self):
        """测试未连接时的健康检查"""
        self.source.connection_status.is_connected = False
        result = self.source.health_check()
        self.assertFalse(result)
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(self.source, 'connect', return_value=True) as mock_connect:
            with patch.object(self.source, 'disconnect') as mock_disconnect:
                with self.source as src:
                    self.assertEqual(src, self.source)
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    @patch('requests.get')
    def test_streaming_data_simulation(self, mock_get):
        """测试流式数据模拟"""
        # 模拟流式响应
        mock_response = Mock()
        mock_response.status_code = 200
        
        # 模拟流式数据行
        stream_lines = [
            b'{"type":"PRICE","instrument":"EUR_USD","time":"2024-01-01T12:00:00.000000Z","bids":[{"price":"1.2050"}],"asks":[{"price":"1.2053"}]}',
            b'{"type":"PRICE","instrument":"EUR_USD","time":"2024-01-01T12:00:01.000000Z","bids":[{"price":"1.2051"}],"asks":[{"price":"1.2054"}]}'
        ]
        
        mock_response.iter_lines.return_value = iter(stream_lines)
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_get.return_value = mock_response
        
        received_data = []
        
        def callback(data):
            received_data.append(data)
            # 收到2个数据后停止流
            if len(received_data) >= 2:
                self.source.stop_streaming()
        
        # 启动流式数据
        self.source.stream_realtime_data(['EUR_USD'], callback, 0.1)
        
        # 等待流式数据处理
        time.sleep(0.3)
        
        # 验证收到数据
        self.assertGreaterEqual(len(received_data), 1)
        for data in received_data:
            self.assertIsInstance(data, MarketData)
            self.assertEqual(data.symbol, 'EUR_USD')


if __name__ == '__main__':
    unittest.main()