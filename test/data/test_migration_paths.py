#!/usr/bin/env python3
"""
迁移路径验证测试

测试从旧API到新API的平滑过渡，确保用户可以逐步迁移代码
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.core.data_manager import DataManager
from src.data.sources.base import DataPeriod, DataSource
from src.utils.config import Config
from download_data import DataDownloader


class TestMigrationPaths:
    """迁移路径验证测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建测试用的模拟数据"""
        return pd.DataFrame({
            'Open': [100.0] * 30,
            'High': [105.0] * 30,
            'Low': [99.0] * 30,
            'Close': [104.0] * 30,
            'Volume': [1000] * 30
        }, index=pd.date_range('2023-01-01', periods=30, freq='D'))
    
    def test_phase1_legacy_string_usage(self, mock_data):
        """阶段1：继续使用传统字符串参数"""
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
            mock_validation_result.records_count = 30
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 用户继续使用旧的字符串方式
                        result = data_manager.get_stock_data('AAPL', period='1y')
                        
                        assert result is not None
                        assert isinstance(result, pd.DataFrame)
                        assert len(result) == 30
                        
                        # 验证调用参数
                        mock_fetch.assert_called_with('AAPL', '1y', '1d')
    
    def test_phase2_gradual_enum_adoption(self, mock_data):
        """阶段2：逐步采用DataPeriod枚举"""
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
            mock_validation_result.records_count = 30
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 用户开始在新代码中使用DataPeriod枚举
                        result_enum = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                        
                        # 但旧代码中的字符串仍然工作
                        result_string = data_manager.get_stock_data('AAPL', period='1y')
                        
                        assert result_enum is not None
                        assert result_string is not None
                        assert isinstance(result_enum, pd.DataFrame)
                        assert isinstance(result_string, pd.DataFrame)
                        
                        # 两种方式都产生相同的结果
                        assert len(result_enum) == len(result_string)
    
    def test_phase3_new_api_features(self, mock_data):
        """阶段3：利用新的API功能"""
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
            mock_validation_result.records_count = 30
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
                 patch.object(data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
                
                mock_fetch.return_value = mock_data
                mock_fetch_range.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 用户开始使用新的日期范围功能
                        result_new = data_manager.get_stock_data_by_date_range(
                            'AAPL', '2023-01-01', '2023-12-31'
                        )
                        
                        # 同时，旧的API仍然工作
                        result_old = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                        
                        assert result_new is not None
                        assert result_old is not None
                        assert isinstance(result_new, pd.DataFrame)
                        assert isinstance(result_old, pd.DataFrame)
    
    def test_mixed_codebase_scenario(self, mock_data):
        """测试混合代码库场景（不同模块使用不同API）"""
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
            mock_validation_result.records_count = 30
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch, \
                 patch.object(data_manager.date_range_fetcher, 'fetch_data_by_date_range') as mock_fetch_range:
                
                mock_fetch.return_value = mock_data
                mock_fetch_range.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        # 模拟不同模块使用不同的API风格
                        
                        # 模块A：使用传统字符串
                        def legacy_module_call():
                            return data_manager.get_stock_data('AAPL', '1y')
                        
                        # 模块B：使用新的枚举
                        def modern_module_call():
                            return data_manager.get_stock_data('AAPL', DataPeriod.YEAR_1)
                        
                        # 模块C：使用日期范围
                        def advanced_module_call():
                            return data_manager.get_stock_data_by_date_range(
                                'AAPL', '2023-01-01', '2023-12-31'
                            )
                        
                        # 所有模块都应该能正常工作
                        result_legacy = legacy_module_call()
                        result_modern = modern_module_call()
                        result_advanced = advanced_module_call()
                        
                        assert result_legacy is not None
                        assert result_modern is not None
                        assert result_advanced is not None
    
    def test_datadownloader_migration_compatibility(self, mock_data):
        """测试DataDownloader的迁移兼容性"""
        downloader = DataDownloader()
        
        with patch.object(downloader.data_manager, 'get_stock_data') as mock_get:
            mock_get.return_value = mock_data
            
            # 测试从字符串到枚举的迁移
            
            # 旧方式：字符串
            result1 = downloader.download_single_stock('AAPL', period='1y', interval='1d')
            assert result1['status'] == 'success'
            
            # 新方式：DataPeriod枚举
            result2 = downloader.download_single_stock('AAPL', period=DataPeriod.YEAR_1, interval='1d')
            assert result2['status'] == 'success'
            
            # 验证两次调用都触发了相同的底层方法
            assert mock_get.call_count == 2
    
    def test_configuration_migration_workflow(self):
        """测试配置迁移工作流"""
        # 测试不同配置迁移阶段
        
        # 阶段1: 使用默认配置（向后兼容）
        config1 = Config()
        assert config1.data.default_period is not None
        
        # 阶段2: 逐步添加新配置选项
        config2 = Config()
        config2.data.default_period = DataPeriod.YEAR_2.value
        config2.data.date_range_priority = 'period'
        
        # 验证新配置生效
        params = config2.data.get_effective_time_params(period='1y')
        assert isinstance(params, dict)
        
        # 阶段3: 使用完整的新配置
        config3 = Config()
        config3.data.default_start_date = '2022-01-01'
        config3.data.default_end_date = '2023-12-31'
        config3.data.date_range_priority = 'date_range'
        
        params3 = config3.data.get_effective_time_params()
        assert 'start_date' in params3 or 'end_date' in params3 or 'period' in params3
    
    def test_period_to_daterange_conversion_migration(self, mock_data):
        """测试周期到日期范围转换的迁移"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 测试DataPeriod到日期范围的转换工具
            
            # 基本转换
            date_range = data_manager.convert_period_to_date_range(DataPeriod.MONTH_1)
            assert date_range is not None
            assert date_range.start_date is not None
            assert date_range.end_date is not None
            assert date_range.duration_days > 0
            
            # 带自定义结束日期的转换
            custom_end_date = datetime(2023, 12, 31)
            date_range_custom = data_manager.convert_period_to_date_range(
                DataPeriod.YEAR_1,
                end_date=custom_end_date
            )
            assert date_range_custom.end_date.date() == custom_end_date.date()
            
            # 字符串周期转换（向后兼容）
            date_range_string = data_manager.convert_period_to_date_range('1y')
            assert date_range_string is not None
    
    def test_error_handling_migration(self, mock_data):
        """测试错误处理在迁移过程中的一致性"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 测试无效周期参数的错误处理
            with pytest.raises(ValueError):
                DataPeriod.from_string('invalid_period_format')
            
            # 测试无效符号的一致性错误处理
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            data_manager.cache_manager.get.return_value = None
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.side_effect = Exception("Network error")
                
                # 新旧API都应该有一致的错误处理
                with pytest.raises(Exception):
                    data_manager.get_stock_data('INVALID_SYMBOL', period='1y')
                
                with pytest.raises(Exception):
                    data_manager.get_stock_data('INVALID_SYMBOL', period=DataPeriod.YEAR_1)
    
    def test_performance_consistency_across_migration(self, mock_data):
        """测试迁移过程中性能的一致性"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟依赖项
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            
            # 测试缓存行为在新旧API中的一致性
            
            # 第一次调用（字符串方式）- 应该缓存结果
            data_manager.cache_manager.get.return_value = None
            data_manager.cache_manager.set.return_value = None
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.records_count = 30
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        result1 = data_manager.get_stock_data('AAPL', period='1y')
                        
                        # 第二次调用（枚举方式）- 应该使用相同的缓存逻辑
                        data_manager.cache_manager.get.return_value = mock_data
                        result2 = data_manager.get_stock_data('AAPL', period=DataPeriod.YEAR_1)
                        
                        assert result1 is not None
                        assert result2 is not None
                        # 验证缓存管理器被调用
                        assert data_manager.cache_manager.get.called
                        assert data_manager.cache_manager.set.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])