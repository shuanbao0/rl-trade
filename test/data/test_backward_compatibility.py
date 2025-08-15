#!/usr/bin/env python3
"""
测试向后兼容性
确保现有的get_stock_data()方法继续工作，以及DataPeriod枚举和日期范围下载功能的向后兼容性
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pytest
from datetime import datetime

from src.data.core.data_manager import DataManager
from src.data.sources.base import DataPeriod, DataSource
from src.utils.config import Config
from download_data import DataDownloader


class TestBackwardCompatibility(unittest.TestCase):
    """向后兼容性测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        
        # 创建模拟的依赖项
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            self.data_manager = DataManager(self.config)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0],
            'High': [102.0, 103.0, 101.0],
            'Low': [99.0, 100.0, 98.0],
            'Close': [101.0, 100.0, 102.0],
            'Volume': [1000, 1100, 900]
        })
    
    def test_get_stock_data_basic_call(self):
        """测试基本的get_stock_data调用"""
        
        # 模拟市场类型检测
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        
        # 模拟缓存未命中
        self.data_manager.cache_manager.get.return_value = None
        
        # 模拟市场类型检测
        self.data_manager.market_detector.detect.return_value = Mock(value='stock')
        
        # 模拟数据获取 - 使用新的服务架构
        with patch.object(self.data_manager.data_fetcher, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = self.test_data
            
            # 模拟数据验证 - 使用新的服务架构
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.records_count = 3
            mock_validation_result.issues = []
            
            with patch.object(self.data_manager.data_validator, 'validate_data') as mock_validate:
                mock_validate.return_value = mock_validation_result
                
                # 模拟数据清洗 - 使用新的服务架构
                with patch.object(self.data_manager.data_validator, 'clean_data') as mock_clean:
                    mock_clean.return_value = self.test_data
                    
                    # 调用旧的API
                    result = self.data_manager.get_stock_data("AAPL")
                    
                    # 验证结果
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, pd.DataFrame)
                    self.assertEqual(len(result), 3)
    
    def test_get_stock_data_with_parameters(self):
        """测试带参数的get_stock_data调用"""
        
        # 模拟市场类型检测
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        
        # 模拟缓存未命中
        self.data_manager.cache_manager.get.return_value = None
        
        # 模拟数据获取
        with patch.object(self.data_manager, '_fetch_from_data_source') as mock_fetch:
            mock_fetch.return_value = self.test_data
            
            # 模拟数据验证和清洗
            with patch.object(self.data_manager, 'validate_data') as mock_validate, \
                 patch.object(self.data_manager, '_clean_data') as mock_clean:
                
                mock_validate.return_value = Mock(
                    is_valid=True,
                    records_count=3,
                    issues=[]
                )
                mock_clean.return_value = self.test_data
                
                # 测试不同参数组合
                result1 = self.data_manager.get_stock_data("AAPL", period="1y")
                result2 = self.data_manager.get_stock_data("AAPL", period="6mo", interval="1h")
                result3 = self.data_manager.get_stock_data("AAPL", force_refresh=True)
                
                # 验证所有调用都成功
                self.assertIsNotNone(result1)
                self.assertIsNotNone(result2)
                self.assertIsNotNone(result3)
    
    def test_get_stock_data_cache_integration(self):
        """测试get_stock_data与新缓存系统的集成"""
        
        # 重置缓存管理器的调用计数
        self.data_manager.cache_manager.reset_mock()
        
        # 模拟市场类型检测
        mock_market_type = Mock()
        mock_market_type.value = 'stock'
        self.data_manager.market_detector.detect_market_type.return_value = mock_market_type
        
        # 模拟缓存命中
        self.data_manager.cache_manager.get.return_value = self.test_data
        
        # 调用get_stock_data
        result = self.data_manager.get_stock_data("AAPL")
        
        # 验证缓存被调用
        self.data_manager.cache_manager.get.assert_called_once()
        
        # 验证返回缓存的数据
        self.assertTrue(result.equals(self.test_data))
    
    def test_get_stock_data_error_handling(self):
        """测试get_stock_data的错误处理兼容性"""
        
        # 模拟市场类型检测
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        
        # 模拟缓存未命中
        self.data_manager.cache_manager.get.return_value = None
        
        # 模拟数据获取失败
        with patch.object(self.data_manager, '_fetch_from_data_source') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            
            # 验证异常被正确传播
            with self.assertRaises(Exception):
                self.data_manager.get_stock_data("AAPL")
    
    def test_new_api_get_market_data(self):
        """测试新API get_market_data的功能"""
        
        # 模拟市场类型检测
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        
        # 模拟缓存未命中
        self.data_manager.cache_manager.get.return_value = None
        
        # 模拟路由管理器
        self.data_manager.routing_manager.get_optimal_sources.return_value = ["yfinance"]
        
        # 模拟数据获取
        with patch.object(self.data_manager, '_fetch_from_single_source') as mock_fetch:
            mock_fetch.return_value = self.test_data
            
            # 模拟数据处理
            with patch.object(self.data_manager, '_process_market_data') as mock_process:
                mock_process.return_value = self.test_data
                
                # 测试新API的不同调用方式
                
                # 1. 不指定数据源（自动选择）
                result1 = self.data_manager.get_market_data("AAPL")
                self.assertIsNotNone(result1)
                
                # 2. 指定数据源
                with patch.object(self.data_manager, '_fetch_from_specified_source') as mock_specified:
                    mock_specified.return_value = self.test_data
                    
                    result2 = self.data_manager.get_market_data("AAPL", source="yfinance")
                    self.assertIsNotNone(result2)
                    mock_specified.assert_called_once()
    
    def test_api_parameter_compatibility(self):
        """测试API参数兼容性"""
        
        # 模拟依赖
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        self.data_manager.cache_manager.get.return_value = None
        
        with patch.object(self.data_manager, '_fetch_from_data_source') as mock_old_fetch, \
             patch.object(self.data_manager, '_fetch_with_routing') as mock_new_fetch:
            
            mock_old_fetch.return_value = self.test_data
            mock_new_fetch.return_value = self.test_data
            
            # 模拟数据验证和清洗
            with patch.object(self.data_manager, 'validate_data') as mock_validate, \
                 patch.object(self.data_manager, '_clean_data') as mock_clean:
                
                mock_validate.return_value = Mock(is_valid=True, records_count=3, issues=[])
                mock_clean.return_value = self.test_data
                
                # 测试旧API的所有参数组合
                old_api_calls = [
                    lambda: self.data_manager.get_stock_data("AAPL"),
                    lambda: self.data_manager.get_stock_data("AAPL", period="1y"),
                    lambda: self.data_manager.get_stock_data("AAPL", period="6mo", interval="1h"),
                    lambda: self.data_manager.get_stock_data("AAPL", force_refresh=True),
                ]
                
                # 测试新API的参数组合
                new_api_calls = [
                    lambda: self.data_manager.get_market_data("AAPL"),
                    lambda: self.data_manager.get_market_data("AAPL", period="1y"),
                    lambda: self.data_manager.get_market_data("AAPL", interval="1h"),
                    lambda: self.data_manager.get_market_data("AAPL", force_refresh=True),
                ]
                
                # 验证所有调用都能正常工作
                for call in old_api_calls:
                    try:
                        result = call()
                        self.assertIsNotNone(result)
                    except Exception as e:
                        self.fail(f"Old API call failed: {e}")
                
                for call in new_api_calls:
                    try:
                        result = call()
                        self.assertIsNotNone(result)
                    except Exception as e:
                        self.fail(f"New API call failed: {e}")
    
    def test_data_format_consistency(self):
        """测试新旧API返回数据格式的一致性"""
        
        # 模拟依赖
        self.data_manager.market_detector.detect_market_type.return_value = Mock(value='stock')
        self.data_manager.cache_manager.get.return_value = None
        
        # 创建标准的股票数据格式
        standard_stock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 99.0],
            'High': [102.0, 103.0, 101.0],
            'Low': [99.0, 100.0, 98.0],
            'Close': [101.0, 100.0, 102.0],
            'Volume': [1000, 1100, 900]
        })
        
        with patch.object(self.data_manager, '_fetch_from_data_source') as mock_old_fetch, \
             patch.object(self.data_manager, '_fetch_with_routing') as mock_new_fetch:
            
            mock_old_fetch.return_value = standard_stock_data
            mock_new_fetch.return_value = standard_stock_data
            
            # 模拟数据验证和清洗
            with patch.object(self.data_manager, 'validate_data') as mock_validate, \
                 patch.object(self.data_manager, '_clean_data') as mock_clean:
                
                mock_validate.return_value = Mock(is_valid=True, records_count=3, issues=[])
                mock_clean.return_value = standard_stock_data
                
                # 获取新旧API的结果
                old_result = self.data_manager.get_stock_data("AAPL")
                new_result = self.data_manager.get_market_data("AAPL")
                
                # 验证数据格式一致性
                self.assertEqual(list(old_result.columns), list(new_result.columns))
                self.assertEqual(len(old_result), len(new_result))
                self.assertEqual(old_result.index.name, new_result.index.name)
                
                # 验证数据类型一致性
                for col in old_result.columns:
                    self.assertEqual(old_result[col].dtype, new_result[col].dtype)


class TestDataPeriodBackwardCompatibility(unittest.TestCase):
    """测试DataPeriod枚举的向后兼容性"""
    
    def test_string_period_parameters(self):
        """测试字符串周期参数的完全兼容性"""
        # 测试所有支持的字符串格式
        string_periods = [
            '1d', '7d', '30d', '60d', '90d',
            '1w', '2w', '4w', 
            '1mo', '3mo', '6mo', '12mo',
            '1y', '2y', '5y', '10y',
            'max'
        ]
        
        for period_str in string_periods:
            with self.subTest(period=period_str):
                # 字符串应该能转换为DataPeriod枚举
                try:
                    period_enum = DataPeriod.from_string(period_str)
                    self.assertIsInstance(period_enum, DataPeriod)
                    self.assertEqual(period_enum.value, period_str)
                except ValueError:
                    # 某些字符串可能不被支持，但这应该是已知的
                    self.fail(f"字符串 '{period_str}' 无法转换为DataPeriod")
    
    def test_datamanager_string_period_compatibility(self):
        """测试DataManager对字符串周期的兼容性"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟数据返回
            mock_data = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [99], 
                'Close': [104], 'Volume': [1000]
            })
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                # 测试字符串参数
                result1 = data_manager.get_stock_data('AAPL', period='1y')
                self.assertIsNotNone(result1)
                
                # 测试枚举参数
                result2 = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                self.assertIsNotNone(result2)
    
    def test_datadownloader_period_compatibility(self):
        """测试DataDownloader对不同period类型的兼容性"""
        downloader = DataDownloader()
        
        with patch.object(downloader.data_manager, 'get_stock_data') as mock_get:
            mock_data = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [99], 
                'Close': [104], 'Volume': [1000]
            })
            mock_get.return_value = mock_data
            
            # 测试字符串参数（旧方式）
            result1 = downloader.download_single_stock('AAPL', period='1y')
            self.assertEqual(result1['status'], 'success')
            
            # 测试DataPeriod枚举（新方式）
            result2 = downloader.download_single_stock('AAPL', period=DataPeriod.YEAR_1)
            self.assertEqual(result2['status'], 'success')


class TestDateRangeBackwardCompatibility(unittest.TestCase):
    """测试日期范围功能的向后兼容性"""
    
    def setUp(self):
        self.config = Config()
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            self.data_manager = DataManager(self.config)
    
    def test_existing_api_unchanged(self):
        """测试现有API保持不变"""
        mock_data = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [99], 
            'Close': [104], 'Volume': [1000]
        })
        
        with patch.object(self.data_manager.data_fetcher, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = mock_data
            
            # 原有的API调用方式应该完全不变
            result = self.data_manager.get_stock_data('AAPL', period='1y', interval='1d')
            self.assertIsNotNone(result)
            
            # 验证调用参数
            mock_fetch.assert_called_with('AAPL', '1y', '1d')
    
    def test_new_date_range_api_coexists(self):
        """测试新的日期范围API与旧API共存"""
        mock_data = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [99], 
            'Close': [104], 'Volume': [1000]
        })
        
        with patch.object(self.data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
             patch.object(self.data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
            
            mock_fetch.return_value = mock_data
            mock_fetch_range.return_value = mock_data
            
            # 旧API
            result1 = self.data_manager.get_stock_data('AAPL', period='1y')
            self.assertIsNotNone(result1)
            
            # 新API
            result2 = self.data_manager.get_stock_data_by_date_range(
                'AAPL', '2023-01-01', '2023-12-31'
            )
            self.assertIsNotNone(result2)
            
            # 两种API都应该被调用
            mock_fetch.assert_called()
            mock_fetch_range.assert_called()
    
    def test_parameter_priority_backward_compatibility(self):
        """测试参数优先级不影响向后兼容性"""
        mock_data = pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [99], 
            'Close': [104], 'Volume': [1000]
        })
        
        with patch.object(self.data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
            mock_fetch_range.return_value = mock_data
            
            # 当同时提供period和日期范围时，应该使用日期范围（新优先级逻辑）
            # 但旧的API调用方式仍然有效
            result = self.data_manager.get_stock_data(
                'AAPL', 
                period='1y',  # 会被忽略
                start_date='2023-01-01',  # 优先使用
                end_date='2023-12-31'     # 优先使用
            )
            
            self.assertIsNotNone(result)
            # 应该调用日期范围方法
            mock_fetch_range.assert_called()


class TestConfigurationBackwardCompatibility(unittest.TestCase):
    """测试配置系统的向后兼容性"""
    
    def test_config_fields_backward_compatibility(self):
        """测试配置字段的向后兼容性"""
        config = Config()
        
        # 测试旧的配置字段仍然存在
        self.assertTrue(hasattr(config.data, 'cache_ttl'))
        self.assertTrue(hasattr(config.data, 'retry_count'))
        self.assertTrue(hasattr(config.data, 'timeout'))
        
        # 测试新的配置字段有默认值
        self.assertTrue(hasattr(config.data, 'default_period'))
        self.assertIsNotNone(config.data.default_period)
    
    def test_config_methods_backward_compatibility(self):
        """测试配置方法的向后兼容性"""
        config = Config()
        
        # 新的方法应该能处理旧的参数组合
        params = config.data.get_effective_time_params(period='1y')
        self.assertIsInstance(params, dict)
        self.assertIn('period', params)
    
    def test_environment_variable_compatibility(self):
        """测试环境变量兼容性"""
        import os
        
        # 临时设置环境变量
        original_value = os.environ.get('DEFAULT_PERIOD')
        os.environ['DEFAULT_PERIOD'] = '2y'
        
        try:
            config = Config()
            # 应该从环境变量读取
            self.assertEqual(config.data.default_period, '2y')
        finally:
            # 恢复原始值
            if original_value is None:
                os.environ.pop('DEFAULT_PERIOD', None)
            else:
                os.environ['DEFAULT_PERIOD'] = original_value


class TestAPIEvolutionCompatibility(unittest.TestCase):
    """测试API演进的兼容性"""
    
    def test_gradual_migration_path(self):
        """测试渐进式迁移路径"""
        config = Config()
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            mock_data = pd.DataFrame({
                'Open': [100], 'High': [105], 'Low': [99], 
                'Close': [104], 'Volume': [1000]
            })
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
                 patch.object(data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
                
                mock_fetch.return_value = mock_data
                mock_fetch_range.return_value = mock_data
                
                # 阶段1: 继续使用字符串
                result1 = data_manager.get_stock_data('AAPL', period='1y')
                self.assertIsNotNone(result1)
                
                # 阶段2: 采用枚举
                result2 = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                self.assertIsNotNone(result2)
                
                # 阶段3: 使用新功能
                result3 = data_manager.get_stock_data_by_date_range(
                    'AAPL', '2023-01-01', '2023-12-31'
                )
                self.assertIsNotNone(result3)
    
    def test_return_format_consistency(self):
        """测试返回格式的一致性"""
        config = Config()
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            mock_data = pd.DataFrame({
                'Open': [100.0, 101.0],
                'High': [105.0, 106.0], 
                'Low': [99.0, 100.0],
                'Close': [104.0, 105.0],
                'Volume': [1000, 1100]
            })
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
                 patch.object(data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
                
                mock_fetch.return_value = mock_data
                mock_fetch_range.return_value = mock_data
                
                # 旧API调用
                old_result = data_manager.get_stock_data('AAPL', '1y')
                
                # 新API调用
                new_result = data_manager.get_stock_data_by_date_range(
                    'AAPL', '2023-01-01', '2023-12-31'
                )
                
                # 验证返回格式一致性
                self.assertEqual(list(old_result.columns), list(new_result.columns))
                self.assertIsInstance(old_result, pd.DataFrame)
                self.assertIsInstance(new_result, pd.DataFrame)


class TestImportPathsCompatibility(unittest.TestCase):
    """测试导入路径的兼容性"""
    
    def test_main_imports_unchanged(self):
        """测试主要导入路径没有改变"""
        # 这些导入应该仍然工作
        try:
            from src.data.core.data_manager import DataManager
            from src.data.sources.base import DataSource, DataPeriod
            from src.utils.config import Config
            
            # 验证类可以实例化
            self.assertIsNotNone(DataManager)
            self.assertIsNotNone(DataSource)
            self.assertIsNotNone(DataPeriod)
            self.assertIsNotNone(Config)
            
        except ImportError as e:
            self.fail(f"导入路径兼容性失败: {e}")
    
    def test_enum_string_interoperability(self):
        """测试枚举和字符串的互操作性"""
        # 枚举到字符串
        enum_period = DataPeriod.YEAR_1
        string_value = enum_period.value
        self.assertEqual(string_value, '1y')
        
        # 字符串到枚举
        converted_back = DataPeriod.from_string(string_value)
        self.assertEqual(converted_back, enum_period)
        
        # 验证属性
        self.assertEqual(enum_period.display_name, "1 Year")
        self.assertEqual(enum_period.to_days(), 365)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)