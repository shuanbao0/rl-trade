"""
测试数据源工厂和注册机制
"""

import pytest
import unittest
import tempfile
import os
from pathlib import Path
import yaml

from src.data.sources.factory import (
    DataSourceRegistry, DataSourceFactory, ConfigLoader
)
from src.data.sources.base import AbstractDataSource, DataSourceCapabilities, MarketType, DataInterval


class MockDataSource1(AbstractDataSource):
    """模拟数据源1"""
    
    def connect(self):
        return True
    
    def disconnect(self):
        pass
    
    def fetch_historical_data(self, symbol, start_date, end_date, interval):
        import pandas as pd
        return pd.DataFrame()
    
    def fetch_realtime_data(self, symbols):
        from src.data.sources.base import MarketData
        from datetime import datetime
        return MarketData(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100, high=100, low=100, close=100, volume=0
        )
    
    def validate_symbol(self, symbol):
        return True
    
    def get_capabilities(self):
        return DataSourceCapabilities(
            name="MockSource1",
            supported_markets=[MarketType.STOCK],
            supported_intervals=[DataInterval.MINUTE_1],
            has_realtime=True,
            has_historical=True
        )


class MockDataSource2(AbstractDataSource):
    """模拟数据源2"""
    
    def connect(self):
        return True
    
    def disconnect(self):
        pass
    
    def fetch_historical_data(self, symbol, start_date, end_date, interval):
        import pandas as pd
        return pd.DataFrame()
    
    def fetch_realtime_data(self, symbols):
        from src.data.sources.base import MarketData
        from datetime import datetime
        return MarketData(
            symbol="TEST",
            timestamp=datetime.now(),
            open=200, high=200, low=200, close=200, volume=0
        )
    
    def validate_symbol(self, symbol):
        return True
    
    def get_capabilities(self):
        return DataSourceCapabilities(
            name="MockSource2",
            supported_markets=[MarketType.FOREX],
            supported_intervals=[DataInterval.HOUR_1],
            has_realtime=True,
            has_historical=True
        )


class TestDataSourceRegistry(unittest.TestCase):
    """测试数据源注册表"""
    
    def setUp(self):
        """设置测试"""
        # 清空注册表
        DataSourceRegistry.clear()
    
    def tearDown(self):
        """清理测试"""
        DataSourceRegistry.clear()
    
    def test_register_source(self):
        """测试注册数据源"""
        # 注册数据源
        DataSourceRegistry.register('mock1', MockDataSource1)
        
        # 验证注册成功
        self.assertTrue(DataSourceRegistry.is_registered('mock1'))
        self.assertIn('mock1', DataSourceRegistry.list_sources())
        
        # 获取数据源类
        source_class = DataSourceRegistry.get('mock1')
        self.assertEqual(source_class, MockDataSource1)
    
    def test_register_with_config(self):
        """测试带配置注册"""
        config = {'param1': 'value1', 'param2': 123}
        DataSourceRegistry.register('mock1', MockDataSource1, config)
        
        # 验证配置
        stored_config = DataSourceRegistry.get_config('mock1')
        self.assertEqual(stored_config, config)
    
    def test_case_insensitive(self):
        """测试大小写不敏感"""
        DataSourceRegistry.register('Mock1', MockDataSource1)
        
        # 不同大小写都应该能找到
        self.assertTrue(DataSourceRegistry.is_registered('mock1'))
        self.assertTrue(DataSourceRegistry.is_registered('MOCK1'))
        self.assertTrue(DataSourceRegistry.is_registered('Mock1'))
        
        source_class = DataSourceRegistry.get('MOCK1')
        self.assertEqual(source_class, MockDataSource1)
    
    def test_unregister_source(self):
        """测试注销数据源"""
        DataSourceRegistry.register('mock1', MockDataSource1)
        self.assertTrue(DataSourceRegistry.is_registered('mock1'))
        
        DataSourceRegistry.unregister('mock1')
        self.assertFalse(DataSourceRegistry.is_registered('mock1'))
    
    def test_multiple_sources(self):
        """测试多个数据源"""
        DataSourceRegistry.register('mock1', MockDataSource1)
        DataSourceRegistry.register('mock2', MockDataSource2)
        
        sources = DataSourceRegistry.list_sources()
        self.assertEqual(set(sources), {'mock1', 'mock2'})
    
    def test_get_source_info(self):
        """测试获取数据源信息"""
        config = {'test': 'value'}
        DataSourceRegistry.register('mock1', MockDataSource1, config)
        
        info = DataSourceRegistry.get_source_info('mock1')
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], 'mock1')
        self.assertEqual(info['class'], 'MockDataSource1')
        self.assertEqual(info['config'], config)
    
    def test_invalid_registration(self):
        """测试无效注册"""
        # 注册非类对象
        with self.assertRaises(TypeError):
            DataSourceRegistry.register('invalid', "not_a_class")
        
        # 注册不是AbstractDataSource子类的类
        class NotDataSource:
            pass
        
        with self.assertRaises(TypeError):
            DataSourceRegistry.register('invalid', NotDataSource)


class TestConfigLoader(unittest.TestCase):
    """测试配置加载器"""
    
    def test_load_yaml(self):
        """测试YAML配置加载"""
        config_data = {
            'type': 'test_source',
            'param1': 'value1',
            'param2': 123,
            'nested': {
                'key1': 'value1',
                'key2': 456
            }
        }
        
        # 创建临时YAML文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_file = f.name
        
        try:
            # 加载配置
            loaded_config = ConfigLoader.load_yaml(yaml_file)
            self.assertEqual(loaded_config, config_data)
        finally:
            os.unlink(yaml_file)
    
    def test_load_json(self):
        """测试JSON配置加载"""
        config_data = {
            'type': 'test_source',
            'param1': 'value1',
            'param2': 123
        }
        
        # 创建临时JSON文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            json_file = f.name
        
        try:
            # 加载配置
            loaded_config = ConfigLoader.load_json(json_file)
            self.assertEqual(loaded_config, config_data)
        finally:
            os.unlink(json_file)
    
    def test_env_var_resolution(self):
        """测试环境变量解析"""
        # 设置测试环境变量
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['TEST_NUM'] = '42'
        
        try:
            config_data = {
                'simple_var': '${TEST_VAR}',
                'var_with_default': '${NONEXISTENT:default_value}',
                'number_var': '${TEST_NUM}',
                'nested': {
                    'env_var': '${TEST_VAR}'
                }
            }
            
            resolved = ConfigLoader._resolve_env_vars(config_data)
            
            self.assertEqual(resolved['simple_var'], 'test_value')
            self.assertEqual(resolved['var_with_default'], 'default_value')
            self.assertEqual(resolved['number_var'], '42')
            self.assertEqual(resolved['nested']['env_var'], 'test_value')
            
        finally:
            del os.environ['TEST_VAR']
            del os.environ['TEST_NUM']
    
    def test_file_not_found(self):
        """测试文件不存在"""
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load_yaml('/nonexistent/file.yaml')
        
        with self.assertRaises(FileNotFoundError):
            ConfigLoader.load_json('/nonexistent/file.json')


class TestDataSourceFactory(unittest.TestCase):
    """测试数据源工厂"""
    
    def setUp(self):
        """设置测试"""
        DataSourceRegistry.clear()
        DataSourceRegistry.register('mock1', MockDataSource1, {'default_param': 'default_value'})
        DataSourceRegistry.register('mock2', MockDataSource2)
    
    def tearDown(self):
        """清理测试"""
        DataSourceRegistry.clear()
    
    def test_create_data_source(self):
        """测试创建数据源"""
        # 创建无配置的数据源
        source = DataSourceFactory.create_data_source('mock1')
        self.assertIsInstance(source, MockDataSource1)
        self.assertEqual(source.name, 'mock1')
        
        # 验证默认配置被应用
        self.assertEqual(source.config['default_param'], 'default_value')
    
    def test_create_with_custom_config(self):
        """测试使用自定义配置创建"""
        config = {'custom_param': 'custom_value', 'param2': 123}
        source = DataSourceFactory.create_data_source('mock1', config)
        
        self.assertIsInstance(source, MockDataSource1)
        # 应该同时包含默认配置和自定义配置
        self.assertEqual(source.config['default_param'], 'default_value')
        self.assertEqual(source.config['custom_param'], 'custom_value')
        self.assertEqual(source.config['param2'], 123)
    
    def test_unknown_source_type(self):
        """测试未知数据源类型"""
        with self.assertRaises(ValueError) as cm:
            DataSourceFactory.create_data_source('unknown_source')
        
        self.assertIn('Unknown data source type', str(cm.exception))
        self.assertIn('unknown_source', str(cm.exception))
    
    def test_invalid_config_type(self):
        """测试无效配置类型"""
        with self.assertRaises(TypeError):
            DataSourceFactory.create_data_source('mock1', "invalid_config")
    
    def test_create_multiple(self):
        """测试批量创建"""
        sources_config = {
            'mock1': {'param1': 'value1'},
            'mock2': {'param2': 'value2'},
            'invalid': {'param': 'value'}  # 这个应该失败
        }
        
        sources = DataSourceFactory.create_multiple(sources_config)
        
        # 应该创建成功的数据源
        self.assertIn('mock1', sources)
        self.assertIn('mock2', sources)
        self.assertIsInstance(sources['mock1'], MockDataSource1)
        self.assertIsInstance(sources['mock2'], MockDataSource2)
        
        # 失败的应该不在结果中
        self.assertNotIn('invalid', sources)
    
    def test_get_default_source(self):
        """测试获取默认数据源"""
        # 当有多个源时，应该按优先级返回
        default = DataSourceFactory.get_default_source()
        self.assertIsNotNone(default)
        self.assertIsInstance(default, AbstractDataSource)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 有效配置
        is_valid, errors = DataSourceFactory.validate_config('mock1', {'param': 'value'})
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # 无效数据源类型
        is_valid, errors = DataSourceFactory.validate_config('invalid_source', {})
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_create_from_config_file(self):
        """测试从配置文件创建"""
        config_data = {
            'type': 'mock1',
            'custom_param': 'file_value'
        }
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            source = DataSourceFactory.create_from_config_file(config_file)
            self.assertIsInstance(source, MockDataSource1)
            self.assertEqual(source.config['custom_param'], 'file_value')
        finally:
            os.unlink(config_file)
    
    def test_create_from_env(self):
        """测试从环境变量创建"""
        # 设置环境变量
        os.environ['DATA_SOURCE_MOCK1_PARAM1'] = 'env_value1'
        os.environ['DATA_SOURCE_MOCK1_PARAM2'] = '42'
        os.environ['DATA_SOURCE_MOCK1_ENABLED'] = 'true'
        
        try:
            source = DataSourceFactory.create_from_env('mock1')
            self.assertIsInstance(source, MockDataSource1)
            self.assertEqual(source.config['param1'], 'env_value1')
            self.assertEqual(source.config['param2'], 42)
            self.assertEqual(source.config['enabled'], True)
        finally:
            # 清理环境变量
            for key in ['DATA_SOURCE_MOCK1_PARAM1', 'DATA_SOURCE_MOCK1_PARAM2', 'DATA_SOURCE_MOCK1_ENABLED']:
                if key in os.environ:
                    del os.environ[key]


if __name__ == '__main__':
    unittest.main()