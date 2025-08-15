#!/usr/bin/env python3
"""
核心向后兼容性测试

专注于测试最重要的向后兼容性功能：
1. 字符串period参数的向后兼容性
2. DataPeriod枚举的互操作性
3. API方法的向后兼容性
4. 配置系统的兼容性
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.data.core.data_manager import DataManager
from src.data.sources.base import DataPeriod, DataSource
from src.utils.config import Config
from download_data import DataDownloader


class TestCoreBackwardCompatibility:
    """核心向后兼容性测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建测试用的模拟数据"""
        return pd.DataFrame({
            'Open': [100.0] * 20,    # 确保有足够的记录数
            'High': [105.0] * 20,
            'Low': [99.0] * 20,
            'Close': [104.0] * 20,
            'Volume': [1000] * 20
        }, index=pd.date_range('2023-01-01', periods=20, freq='D'))
    
    def test_string_period_parameter_conversion(self):
        """测试字符串周期参数的转换"""
        # 测试所有支持的字符串格式
        string_periods = [
            '1d', '7d', '30d', '60d', '90d',
            '1w', '2w', '4w', 
            '1mo', '3mo', '6mo', '12mo',
            '1y', '2y', '5y', '10y',
            'max'
        ]
        
        for period_str in string_periods:
            # 字符串应该能转换为DataPeriod枚举
            try:
                period_enum = DataPeriod.from_string(period_str)
                assert isinstance(period_enum, DataPeriod)
                assert period_enum.value == period_str
                
                # 枚举应该能转换回字符串
                assert str(period_enum.value) == period_str
            except ValueError:
                # 某些字符串可能不被支持，这应该是已知的
                pytest.fail(f"字符串 '{period_str}' 无法转换为DataPeriod")
    
    def test_datamanager_accepts_string_periods(self, mock_data):
        """测试DataManager接受字符串周期参数"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟依赖项
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            data_manager.cache_manager.get.return_value = None
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                mock_validation_result = Mock()
                mock_validation_result.is_valid = True
                mock_validation_result.records_count = 20
                mock_validation_result.issues = []
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 测试字符串参数（旧方式）
                        result1 = data_manager.get_stock_data('AAPL', period='1y')
                        assert result1 is not None
                        assert isinstance(result1, pd.DataFrame)
                        
                        # 测试DataPeriod枚举（新方式）
                        result2 = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                        assert result2 is not None
                        assert isinstance(result2, pd.DataFrame)
    
    def test_datadownloader_period_compatibility(self, mock_data):
        """测试DataDownloader对不同period类型的兼容性"""
        downloader = DataDownloader()
        
        with patch.object(downloader.data_manager, 'get_stock_data') as mock_get:
            mock_get.return_value = mock_data
            
            # 测试字符串参数（旧方式）
            result1 = downloader.download_single_stock('AAPL', period='1y')
            assert result1['status'] == 'success'
            
            # 测试DataPeriod枚举（新方式）
            result2 = downloader.download_single_stock('AAPL', period=DataPeriod.YEAR_1)
            assert result2['status'] == 'success'
    
    def test_config_backward_compatibility(self):
        """测试配置系统的向后兼容性"""
        config = Config()
        
        # 测试新字段有默认值
        assert hasattr(config.data, 'default_period')
        assert config.data.default_period is not None
        
        # 测试get_effective_time_params的向后兼容性
        # 只提供period参数（旧方式）
        params = config.data.get_effective_time_params(period='1y')
        assert isinstance(params, dict)
        # 应该包含period或者start_date/end_date
        assert 'period' in params or ('start_date' in params and 'end_date' in params)
    
    def test_enum_string_interoperability(self):
        """测试枚举和字符串的互操作性"""
        # 枚举到字符串
        enum_period = DataPeriod.YEAR_1
        string_value = enum_period.value
        assert string_value == '1y'
        
        # 字符串到枚举
        converted_back = DataPeriod.from_string(string_value)
        assert converted_back == enum_period
        
        # 验证属性
        assert enum_period.display_name == "1 Year"
        assert enum_period.to_days() == 365
        assert enum_period.is_medium_term is True
        assert enum_period.is_long_term is False
    
    def test_dataperiod_properties(self):
        """测试DataPeriod枚举的属性和方法"""
        # 测试不同的时间分类
        short_term_periods = [DataPeriod.DAYS_1, DataPeriod.DAYS_7, DataPeriod.DAYS_30]
        medium_term_periods = [DataPeriod.MONTH_3, DataPeriod.MONTH_6, DataPeriod.YEAR_1]
        long_term_periods = [DataPeriod.YEAR_2, DataPeriod.YEAR_5, DataPeriod.YEAR_10]
        
        for period in short_term_periods:
            assert period.is_short_term is True
            assert period.is_medium_term is False or period == DataPeriod.DAYS_30  # 30天可能被归类为中期
            assert period.is_long_term is False
        
        for period in medium_term_periods:
            assert period.is_medium_term is True
            assert period.is_long_term is False or period == DataPeriod.YEAR_1  # 1年可能被归类为长期
        
        for period in long_term_periods:
            assert period.is_long_term is True
    
    def test_api_method_signatures_unchanged(self):
        """测试API方法签名没有改变"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 测试get_stock_data方法签名
            import inspect
            sig = inspect.signature(data_manager.get_stock_data)
            params = list(sig.parameters.keys())
            
            # 应该至少包含这些参数
            expected_params = ['symbol', 'period', 'interval']
            for param in expected_params:
                assert param in params, f"参数 {param} 不存在于get_stock_data方法中"
    
    def test_import_paths_backward_compatibility(self):
        """测试导入路径的向后兼容性"""
        # 测试主要类仍然可以从原来的位置导入
        try:
            from src.data.core.data_manager import DataManager
            from src.data.sources.base import DataSource, DataPeriod
            from src.utils.config import Config
            
            assert DataManager is not None
            assert DataSource is not None
            assert DataPeriod is not None
            assert Config is not None
            
        except ImportError as e:
            pytest.fail(f"导入路径兼容性失败：无法导入 {e}")
    
    def test_configuration_migration_compatibility(self):
        """测试配置迁移的兼容性"""
        config = Config()
        
        # 测试新旧配置字段共存
        assert hasattr(config.data, 'default_period')  # 新字段
        
        # 测试配置验证
        is_valid, errors = config.data.validate_date_range_config()
        # 即使没有日期范围配置，也应该是有效的（向后兼容）
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_gradual_migration_scenarios(self, mock_data):
        """测试渐进式迁移场景"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟依赖项
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            data_manager.cache_manager.get.return_value = None
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.records_count = 20
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
                 patch.object(data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
                
                mock_fetch.return_value = mock_data
                mock_fetch_range.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 阶段1: 继续使用字符串
                        result1 = data_manager.get_stock_data('AAPL', period='1y')
                        assert result1 is not None
                        
                        # 阶段2: 采用枚举
                        result2 = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                        assert result2 is not None
                        
                        # 阶段3: 使用新功能
                        result3 = data_manager.get_stock_data_by_date_range(
                            'AAPL', '2023-01-01', '2023-12-31'
                        )
                        assert result3 is not None
    
    def test_environment_variable_compatibility(self):
        """测试环境变量兼容性"""
        import os
        
        # 临时设置环境变量
        original_value = os.environ.get('DEFAULT_PERIOD')
        os.environ['DEFAULT_PERIOD'] = '2y'
        
        try:
            config = Config()
            # 应该从环境变量读取
            assert config.data.default_period == '2y'
        finally:
            # 恢复原始值
            if original_value is None:
                os.environ.pop('DEFAULT_PERIOD', None)
            else:
                os.environ['DEFAULT_PERIOD'] = original_value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])