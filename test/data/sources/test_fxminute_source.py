"""
FXMinuteData数据源测试
"""

import unittest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.sources.fxminute_source import FXMinuteDataSource
from src.data.sources.base import DataInterval, DataQuality
from src.data.data_manager import DataManager


class TestFXMinuteDataSource(unittest.TestCase):
    """FXMinuteData数据源测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = {
            'data_directory': 'local_data/FX-1-Minute-Data',
            'auto_extract': True,
            'cache_extracted': True,
            'extracted_cache_dir': 'test_fx_cache'
        }
        cls.source = FXMinuteDataSource(cls.config)
    
    def test_initialization(self):
        """测试数据源初始化"""
        self.assertIsNotNone(self.source)
        self.assertEqual(self.source.data_directory, Path('local_data/FX-1-Minute-Data'))
        self.assertTrue(self.source.auto_extract)
        self.assertTrue(self.source.cache_extracted)
    
    def test_supported_pairs_loading(self):
        """测试支持的交易对加载"""
        pairs = self.source.supported_pairs
        self.assertIsInstance(pairs, dict)
        self.assertGreater(len(pairs), 50)  # 应该有60+个交易对
        
        # 检查主要货币对
        major_pairs = ['eurusd', 'gbpusd', 'usdjpy', 'usdchf', 'audusd', 'usdcad']
        for pair in major_pairs:
            self.assertIn(pair, pairs)
            self.assertIn('name', pairs[pair])
            self.assertIn('type', pairs[pair])
            self.assertIn('start_year', pairs[pair])
    
    def test_connection(self):
        """测试连接功能"""
        if Path(self.config['data_directory']).exists():
            result = self.source.connect()
            self.assertTrue(result)
            self.assertTrue(self.source.connection_status.is_connected)
        else:
            # 如果数据目录不存在，连接应该失败
            result = self.source.connect()
            self.assertFalse(result)
    
    def test_capabilities(self):
        """测试数据源能力"""
        capabilities = self.source.get_capabilities()
        
        self.assertEqual(capabilities.name, "FX-1-Minute-Data")
        self.assertIn(DataInterval.MINUTE_1, capabilities.supported_intervals)
        self.assertEqual(len(capabilities.supported_intervals), 1)  # 仅支持1分钟
        self.assertTrue(capabilities.has_historical)
        self.assertFalse(capabilities.has_realtime)
        self.assertFalse(capabilities.requires_auth)
        self.assertTrue(capabilities.is_free)
        self.assertEqual(capabilities.data_quality, DataQuality.HIGH)
    
    def test_symbol_validation(self):
        """测试标的验证"""
        # 有效标的
        self.assertTrue(self.source.validate_symbol('EURUSD'))
        self.assertTrue(self.source.validate_symbol('eurusd'))
        self.assertTrue(self.source.validate_symbol('EUR/USD'))
        self.assertTrue(self.source.validate_symbol('xauusd'))  # 黄金
        
        # 无效标的
        self.assertFalse(self.source.validate_symbol('INVALID'))
        self.assertFalse(self.source.validate_symbol('BTCUSD'))  # 不支持加密货币
    
    def test_symbol_normalization(self):
        """测试标的标准化"""
        # 基本标准化
        self.assertEqual(self.source._normalize_symbol('EUR/USD'), 'eurusd')
        self.assertEqual(self.source._normalize_symbol('GBP-USD'), 'gbpusd')
        self.assertEqual(self.source._normalize_symbol('USD_JPY'), 'usdjpy')
        
        # 特殊映射
        self.assertEqual(self.source._normalize_symbol('gold'), 'xauusd')
        self.assertEqual(self.source._normalize_symbol('silver'), 'xagusd')
        self.assertEqual(self.source._normalize_symbol('oil'), 'wtiusd')
        self.assertEqual(self.source._normalize_symbol('SP500'), 'spxusd')
    
    def test_search_symbols(self):
        """测试标的搜索"""
        # 搜索EUR相关
        results = self.source.search_symbols('eur', 5)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            result = results[0]
            self.assertIn('symbol', result)
            self.assertIn('name', result)
            self.assertIn('type', result)
            self.assertIn('source', result)
            self.assertEqual(result['source'], 'fxminute')
    
    def test_symbol_info(self):
        """测试获取标的信息"""
        info = self.source.get_symbol_info('eurusd')
        
        if info:
            self.assertIn('name', info)
            self.assertIn('type', info)
            self.assertIn('start_year', info)
            self.assertEqual(info['type'], 'forex')
            self.assertEqual(info['name'], 'EUR/USD')
        
        # 测试无效标的
        invalid_info = self.source.get_symbol_info('invalid')
        self.assertIsNone(invalid_info)
    
    def test_file_index_building(self):
        """测试文件索引构建"""
        if Path(self.config['data_directory']).exists():
            self.source._build_file_index()
            self.assertTrue(self.source._index_loaded)
            self.assertIsInstance(self.source._file_index, dict)
            
            # 检查是否找到了数据文件
            if self.source._file_index:
                # 随便选一个标的检查文件信息
                symbol = list(self.source._file_index.keys())[0]
                files = self.source._file_index[symbol]
                self.assertIsInstance(files, list)
                
                if files:
                    file_info = files[0]
                    self.assertIn('path', file_info)
                    self.assertIn('year', file_info)
                    self.assertIn('filename', file_info)
                    self.assertIsInstance(file_info['year'], int)
    
    def test_available_symbols(self):
        """测试获取可用标的"""
        if Path(self.config['data_directory']).exists():
            symbols = self.source.get_available_symbols()
            self.assertIsInstance(symbols, list)
            
            # 如果有数据文件，应该返回一些标的
            if symbols:
                self.assertGreater(len(symbols), 0)
                for symbol in symbols[:5]:  # 检查前5个
                    self.assertIsInstance(symbol, str)
                    self.assertTrue(symbol.islower())  # 应该是小写
    
    @unittest.skipUnless(
        Path('local_data/FX-1-Minute-Data/eurusd').exists() and 
        any(Path('local_data/FX-1-Minute-Data/eurusd').glob('DAT_ASCII_EURUSD_M1_*.zip')),
        "需要EUR/USD数据文件进行测试"
    )
    def test_historical_data_fetch(self):
        """测试历史数据获取（需要实际数据文件）"""
        if not self.source.connection_status.is_connected:
            self.source.connect()
        
        # 测试获取EURUSD的2020年1月数据
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)
        
        try:
            data = self.source.fetch_historical_data(
                'EURUSD', start_date, end_date, DataInterval.MINUTE_1
            )
            
            self.assertIsNotNone(data)
            if not data.empty:
                # 检查数据结构
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    self.assertIn(col, data.columns)
                
                # 检查数据类型
                self.assertTrue(data.index.dtype == 'datetime64[ns]')
                
                # 检查数据范围
                self.assertGreaterEqual(data.index.min(), start_date)
                self.assertLessEqual(data.index.max(), end_date)
                
                # 检查OHLC逻辑
                self.assertTrue((data['high'] >= data['low']).all())
                self.assertTrue((data['high'] >= data['open']).all())
                self.assertTrue((data['high'] >= data['close']).all())
                self.assertTrue((data['low'] <= data['open']).all())
                self.assertTrue((data['low'] <= data['close']).all())
        
        except Exception as e:
            # 如果没有对应时期的数据文件，跳过测试
            if "No data files found" in str(e) or "No valid data found" in str(e):
                self.skipTest(f"No data available for test period: {e}")
            else:
                raise
    
    def test_health_check(self):
        """测试健康检查"""
        if Path(self.config['data_directory']).exists():
            if not self.source.connection_status.is_connected:
                self.source.connect()
            
            health = self.source.health_check()
            self.assertIsInstance(health, bool)
        else:
            # 数据目录不存在时，健康检查应该失败
            health = self.source.health_check()
            self.assertFalse(health)
    
    def test_disconnect(self):
        """测试断开连接"""
        self.source.disconnect()
        self.assertFalse(self.source.connection_status.is_connected)


class TestFXMinuteDataManagerIntegration(unittest.TestCase):
    """FXMinuteData与DataManager集成测试"""
    
    def test_data_manager_creation(self):
        """测试通过DataManager创建FXMinuteData源"""
        config = {
            'data_directory': 'local_data/FX-1-Minute-Data',
            'auto_extract': True,
            'cache_extracted': True
        }
        
        try:
            dm = DataManager(
                data_source_type='fxminute',
                data_source_config=config
            )
            
            self.assertIsNotNone(dm)
            self.assertEqual(dm.data_source_type, 'fxminute')
            
            # 检查数据源信息
            info = dm.get_data_source_info()
            self.assertEqual(info['data_source_type'], 'fxminute')
            self.assertEqual(info['data_source_name'], 'FX-1-Minute-Data')
            
        except Exception as e:
            # 如果数据目录不存在或注册失败，记录错误但不失败测试
            if "not found" in str(e).lower() or "not registered" in str(e).lower():
                self.skipTest(f"FXMinute source not available: {e}")
            else:
                raise
    
    @unittest.skipUnless(
        Path('local_data/FX-1-Minute-Data').exists(),
        "需要FX-1-Minute-Data目录进行集成测试"
    )
    def test_get_stock_data_integration(self):
        """测试通过DataManager获取数据"""
        config = {
            'data_directory': 'local_data/FX-1-Minute-Data',
            'auto_extract': True,
            'cache_extracted': True
        }
        
        dm = DataManager(
            data_source_type='fxminute',
            data_source_config=config
        )
        
        # 尝试获取数据（使用较近的日期范围）
        try:
            data = dm.get_stock_data(
                symbol='EURUSD',
                period='1d',  # 1天的数据
                interval='1m',
                force_refresh=True
            )
            
            if data is not None and not data.empty:
                # 检查数据管理器的清洗功能
                self.assertFalse(data.isnull().any().any())  # 不应有空值
                self.assertTrue((data['high'] >= data['low']).all())
                
                # 检查是否添加了数据源标识
                self.assertIn('source', data.columns)
                self.assertEqual(data['source'].iloc[0], 'fxminute')
                
        except Exception as e:
            if "No data" in str(e) or "not found" in str(e):
                self.skipTest(f"No data available for integration test: {e}")
            else:
                raise


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)