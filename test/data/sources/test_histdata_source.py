"""
测试HistData数据源实现
"""

import pytest
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.sources.histdata_source import HistDataDataSource
from src.data.sources.base import DataInterval, MarketData, DataQuality, MarketType


class TestHistDataDataSource(unittest.TestCase):
    """测试HistData数据源"""
    
    def setUp(self):
        """设置测试"""
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.config = {
            'name': 'test_histdata',
            'data_directory': str(self.temp_dir),
            'auto_download': False,
            'timeout': 10,
            'format': 'generic_ascii'
        }
        self.source = HistDataDataSource(self.config)
        
        # 自动下载配置
        self.auto_config = self.config.copy()
        self.auto_config['auto_download'] = True
    
    def tearDown(self):
        """清理测试"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.source.name, 'test_histdata')
        self.assertEqual(self.source.data_directory, self.temp_dir)
        self.assertFalse(self.source.auto_download)
        self.assertEqual(self.source.timeout, 10)
        self.assertEqual(self.source.default_format, 'generic_ascii')
        self.assertEqual(len(self.source.SUPPORTED_PAIRS), 28)
        self.assertIn('EURUSD', self.source.SUPPORTED_PAIRS)
        self.assertIn('XAUUSD', self.source.SUPPORTED_PAIRS)
    
    def test_connection_success(self):
        """测试成功连接"""
        result = self.source.connect()
        
        self.assertTrue(result)
        self.assertTrue(self.source.connection_status.is_connected)
        self.assertIsNotNone(self.source.connection_status.connected_at)
        self.assertIsNone(self.source.connection_status.last_error)
    
    @patch('requests.head')
    def test_connection_with_auto_download(self, mock_head):
        """测试启用自动下载的连接"""
        mock_head.return_value = Mock(status_code=200)
        
        auto_source = HistDataDataSource(self.auto_config)
        result = auto_source.connect()
        
        self.assertTrue(result)
        mock_head.assert_called_once()
    
    @patch('requests.head')
    def test_connection_failure_auto_download(self, mock_head):
        """测试自动下载连接失败"""
        mock_head.return_value = Mock(status_code=404)
        
        auto_source = HistDataDataSource(self.auto_config)
        result = auto_source.connect()
        
        self.assertFalse(result)
        self.assertFalse(auto_source.connection_status.is_connected)
        self.assertIsNotNone(auto_source.connection_status.last_error)
    
    def test_disconnect(self):
        """测试断开连接"""
        self.source.connection_status.is_connected = True
        self.source.disconnect()
        
        self.assertFalse(self.source.connection_status.is_connected)
    
    def test_symbol_normalization(self):
        """测试货币对标准化"""
        test_cases = [
            ('EUR/USD', 'EURUSD'),
            ('EUR-USD', 'EURUSD'),
            ('EUR_USD', 'EURUSD'),
            ('eur.usd', 'EURUSD'),
            ('eurusd', 'EURUSD'),
            ('XAUUSD', 'XAUUSD'),
            ('XAU/USD', 'XAUUSD'),
            ('GBPJPY', 'GBPJPY'),
            ('UNKNOWN', 'UNKNOWN')
        ]
        
        for input_symbol, expected in test_cases:
            result = self.source._normalize_symbol(input_symbol)
            self.assertEqual(result, expected, f"Failed for {input_symbol}")
    
    def test_validate_symbol(self):
        """测试货币对验证"""
        # 有效货币对
        self.assertTrue(self.source.validate_symbol('EURUSD'))
        self.assertTrue(self.source.validate_symbol('EUR/USD'))
        self.assertTrue(self.source.validate_symbol('GBPUSD'))
        self.assertTrue(self.source.validate_symbol('XAUUSD'))
        
        # 无效货币对
        self.assertFalse(self.source.validate_symbol('INVALID'))
        self.assertFalse(self.source.validate_symbol('ABCDEF'))
    
    def test_search_symbols(self):
        """测试货币对搜索"""
        # 搜索EUR
        results = self.source.search_symbols('EUR', limit=5)
        self.assertLessEqual(len(results), 5)
        self.assertTrue(any('EUR' in r['symbol'] for r in results))
        self.assertTrue(all(r['source'] == 'histdata' for r in results))
        
        # 搜索USD
        results = self.source.search_symbols('USD', limit=10)
        self.assertLessEqual(len(results), 10)
        self.assertTrue(any('USD' in r['symbol'] for r in results))
        
        # 搜索XAU (黄金)
        results = self.source.search_symbols('XAU')
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]['symbol'], 'XAUUSD')
        self.assertEqual(results[0]['name'], 'Gold Spot')
        self.assertEqual(results[0]['type'], 'commodity')
        
        # 搜索不存在的
        results = self.source.search_symbols('XYZ')
        self.assertEqual(len(results), 0)
    
    def test_get_capabilities(self):
        """测试能力获取"""
        caps = self.source.get_capabilities()
        
        self.assertEqual(caps.name, "HistData")
        self.assertEqual(set(caps.supported_markets), {MarketType.FOREX, MarketType.COMMODITIES})
        self.assertEqual(set(caps.supported_intervals), {DataInterval.TICK, DataInterval.MINUTE_1})
        self.assertFalse(caps.has_realtime)
        self.assertTrue(caps.has_historical)
        self.assertFalse(caps.has_streaming)
        self.assertFalse(caps.requires_auth)
        self.assertTrue(caps.is_free)
        self.assertEqual(caps.min_interval, DataInterval.TICK)
        self.assertEqual(caps.max_symbols_per_request, 1)
        self.assertEqual(caps.data_quality, DataQuality.HIGH)
        self.assertEqual(caps.max_history_days, 9000)
    
    def test_parse_filename(self):
        """测试文件名解析"""
        test_cases = [
            ('EURUSD_2024_01.csv', {
                'symbol': 'EURUSD',
                'data_type': 'M1',
                'year': 2024,
                'month': 1,
                'date': datetime(2024, 1, 1)
            }),
            ('GBPUSD_TICK_2023_12.csv', {
                'symbol': 'GBPUSD',
                'data_type': 'tick',
                'year': 2023,
                'month': 12,
                'date': datetime(2023, 12, 1)
            }),
            ('XAUUSD_2022_06.csv', {
                'symbol': 'XAUUSD',
                'data_type': 'M1',
                'year': 2022,
                'month': 6,
                'date': datetime(2022, 6, 1)
            }),
            ('invalid_filename.csv', None)
        ]
        
        for filename, expected in test_cases:
            file_path = Path(filename)
            result = self.source._parse_filename(file_path)
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertIsNotNone(result)
                for key, value in expected.items():
                    self.assertEqual(result[key], value)
    
    def _create_test_m1_file(self, filename: str, start_date: datetime, num_records: int = 100):
        """创建测试用的M1数据文件"""
        file_path = self.temp_dir / filename
        
        # 生成测试数据
        timestamps = []
        data = []
        current_time = start_date
        price = 1.2000
        
        for i in range(num_records):
            # HistData M1格式: YYYYMMDD HHMMSS,OPEN,HIGH,LOW,CLOSE,VOLUME
            timestamp_str = current_time.strftime('%Y%m%d %H%M%S')
            
            # 生成有效的OHLC数据
            open_price = price + np.random.uniform(-0.001, 0.001)
            close_price = open_price + np.random.uniform(-0.001, 0.001)
            
            # 确保高低价的有效性
            max_oc = max(open_price, close_price)
            min_oc = min(open_price, close_price)
            high_price = max_oc + np.random.uniform(0, 0.002)
            low_price = min_oc - np.random.uniform(0, 0.002)
            volume = np.random.randint(0, 1000)
            
            data.append(f"{timestamp_str},{open_price:.5f},{high_price:.5f},{low_price:.5f},{close_price:.5f},{volume}")
            
            current_time += timedelta(minutes=1)
            price = close_price
        
        # 写入文件
        with open(file_path, 'w') as f:
            f.write('\n'.join(data))
        
        return file_path
    
    def _create_test_tick_file(self, filename: str, start_date: datetime, num_records: int = 50):
        """创建测试用的Tick数据文件"""
        file_path = self.temp_dir / filename
        
        # 生成测试数据
        data = []
        current_time = start_date
        bid_price = 1.2000
        spread = 0.0003
        
        for i in range(num_records):
            # HistData Tick格式: YYYYMMDD HHMMSS,BID,ASK
            timestamp_str = current_time.strftime('%Y%m%d %H%M%S')
            
            bid = bid_price + np.random.uniform(-0.0005, 0.0005)
            ask = bid + spread + np.random.uniform(-0.0001, 0.0001)
            
            data.append(f"{timestamp_str},{bid:.5f},{ask:.5f}")
            
            current_time += timedelta(seconds=1)
            bid_price = bid
        
        # 写入文件
        with open(file_path, 'w') as f:
            f.write('\n'.join(data))
        
        return file_path
    
    def test_read_m1_file(self):
        """测试读取M1文件"""
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        file_path = self._create_test_m1_file('EURUSD_2024_01.csv', start_date, 10)
        
        df = self.source._read_m1_file(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
        
        # 检查时间索引
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.index[0], start_date)
    
    def test_read_tick_file(self):
        """测试读取Tick文件"""
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        file_path = self._create_test_tick_file('EURUSD_TICK_2024_01.csv', start_date, 10)
        
        df = self.source._read_tick_file(file_path)
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10)
        self.assertIn('bid', df.columns)
        self.assertIn('ask', df.columns)
        self.assertIn('close', df.columns)  # 中间价
        self.assertIn('spread', df.columns)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('volume', df.columns)
        
        # 检查计算结果
        first_row = df.iloc[0]
        expected_close = (first_row['bid'] + first_row['ask']) / 2
        expected_spread = first_row['ask'] - first_row['bid']
        
        self.assertAlmostEqual(first_row['close'], expected_close, places=5)
        self.assertAlmostEqual(first_row['spread'], expected_spread, places=5)
        self.assertEqual(first_row['open'], first_row['close'])  # Tick数据特点
        self.assertEqual(first_row['volume'], 0)  # Tick数据通常无成交量
    
    def test_scan_available_data(self):
        """测试扫描可用数据"""
        # 创建几个测试文件
        self._create_test_m1_file('EURUSD_2024_01.csv', datetime(2024, 1, 1), 10)
        self._create_test_m1_file('GBPUSD_2024_02.csv', datetime(2024, 2, 1), 10)
        self._create_test_tick_file('EURUSD_TICK_2024_01.csv', datetime(2024, 1, 1), 10)
        
        # 重新扫描
        self.source._scan_available_data()
        
        # 验证结果
        self.assertIn('EURUSD', self.source._available_data)
        self.assertIn('GBPUSD', self.source._available_data)
        
        # EURUSD应该有两个文件（M1和Tick）
        eurusd_files = self.source._available_data['EURUSD']
        self.assertEqual(len(eurusd_files), 2)
        
        # 检查文件信息
        data_types = [f['info']['data_type'] for f in eurusd_files]
        self.assertIn('M1', data_types)
        self.assertIn('tick', data_types)
    
    def test_get_data_files(self):
        """测试获取数据文件"""
        # 创建测试文件
        self._create_test_m1_file('EURUSD_2024_01.csv', datetime(2024, 1, 1), 10)
        self._create_test_m1_file('EURUSD_2024_02.csv', datetime(2024, 2, 1), 10)
        self._create_test_m1_file('EURUSD_2024_03.csv', datetime(2024, 3, 1), 10)
        
        # 扫描数据
        self.source._scan_available_data()
        
        # 测试时间范围查询
        start_date = datetime(2024, 1, 15)
        end_date = datetime(2024, 2, 15)
        
        files = self.source._get_data_files('EURUSD', start_date, end_date, 'M1')
        
        # 应该返回1月和2月的文件
        self.assertEqual(len(files), 2)
        filenames = [f.name for f in files]
        self.assertIn('EURUSD_2024_01.csv', filenames)
        self.assertIn('EURUSD_2024_02.csv', filenames)
        self.assertNotIn('EURUSD_2024_03.csv', filenames)
    
    def test_fetch_historical_data(self):
        """测试历史数据获取"""
        # 创建测试文件
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        self._create_test_m1_file('EURUSD_2024_01.csv', start_date, 100)
        
        # 扫描数据
        self.source.connect()
        
        # 获取历史数据
        query_start = datetime(2024, 1, 1, 10, 30, 0)
        query_end = datetime(2024, 1, 1, 11, 30, 0)
        
        df = self.source.fetch_historical_data(
            'EUR/USD', query_start, query_end, DataInterval.MINUTE_1
        )
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('symbol', df.columns)
        self.assertIn('source', df.columns)
        self.assertEqual(df['symbol'].iloc[0], 'EURUSD')
        self.assertEqual(df['source'].iloc[0], 'histdata')
        
        # 检查时间过滤
        self.assertTrue(df.index.min() >= query_start)
        self.assertTrue(df.index.max() <= query_end)
    
    def test_fetch_historical_data_unsupported_interval(self):
        """测试不支持的时间间隔"""
        with self.assertRaises(ValueError) as context:
            self.source.fetch_historical_data(
                'EURUSD', '2024-01-01', '2024-01-02', DataInterval.HOUR_1
            )
        self.assertIn('Unsupported interval', str(context.exception))
    
    def test_fetch_historical_data_no_files(self):
        """测试没有数据文件的情况"""
        self.source.connect()
        
        with self.assertRaises(FileNotFoundError) as context:
            self.source.fetch_historical_data(
                'EURUSD', '2024-01-01', '2024-01-02', DataInterval.MINUTE_1
            )
        self.assertIn('No data files found', str(context.exception))
    
    def test_fetch_realtime_data(self):
        """测试实时数据获取（实际是最新历史数据）"""
        # 创建测试文件
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        self._create_test_m1_file('EURUSD_2024_01.csv', start_date, 50)
        
        # 扫描数据
        self.source.connect()
        
        # 获取"实时"数据
        result = self.source.fetch_realtime_data('EURUSD')
        
        self.assertIsInstance(result, MarketData)
        self.assertEqual(result.symbol, 'EURUSD')
        self.assertEqual(result.source, 'histdata')
        self.assertEqual(result.quality, DataQuality.MEDIUM)
        self.assertTrue(result.metadata['is_historical'])
    
    def test_fetch_realtime_data_multiple(self):
        """测试多个货币对实时数据获取"""
        # 创建测试文件
        start_date = datetime(2024, 1, 1, 10, 0, 0)
        self._create_test_m1_file('EURUSD_2024_01.csv', start_date, 50)
        self._create_test_m1_file('GBPUSD_2024_01.csv', start_date, 50)
        
        # 扫描数据
        self.source.connect()
        
        # 获取多个货币对数据
        results = self.source.fetch_realtime_data(['EURUSD', 'GBPUSD'])
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        symbols = [r.symbol for r in results]
        self.assertIn('EURUSD', symbols)
        self.assertIn('GBPUSD', symbols)
    
    def test_fetch_realtime_data_no_data(self):
        """测试没有数据时的实时数据获取"""
        self.source.connect()
        
        with self.assertRaises(ValueError) as context:
            self.source.fetch_realtime_data('EURUSD')
        self.assertIn('No recent data available', str(context.exception))
    
    def test_health_check(self):
        """测试健康检查"""
        # 未连接时应该失败
        self.assertFalse(self.source.health_check())
        
        # 连接后应该成功
        self.source.connect()
        self.assertTrue(self.source.health_check())
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(self.source, 'connect', return_value=True) as mock_connect:
            with patch.object(self.source, 'disconnect') as mock_disconnect:
                with self.source as src:
                    self.assertEqual(src, self.source)
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    def test_auto_download_not_implemented(self):
        """测试自动下载未实现"""
        auto_source = HistDataDataSource(self.auto_config)
        
        with self.assertRaises(NotImplementedError) as context:
            auto_source._download_data('EURUSD', datetime(2024, 1, 1), datetime(2024, 1, 2), 'M1')
        
        self.assertIn('Automatic download from HistData is not implemented', str(context.exception))


if __name__ == '__main__':
    unittest.main()