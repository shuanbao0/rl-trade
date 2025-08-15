#!/usr/bin/env python3
"""
性能优化验证测试

验证时间参数功能的性能优化特性
"""

import pytest
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.data.core.data_manager import DataManager
from src.data.sources.base import DataPeriod, DataSource
from src.data.advisors.smart_time_advisor import get_smart_time_advisor
from src.utils.config import Config


class TestPerformanceOptimization:
    """性能优化验证测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建测试用的模拟数据"""
        return pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [105.0] * 50,
            'Low': [99.0] * 50,
            'Close': [104.0] * 50,
            'Volume': [1000] * 50
        }, index=pd.date_range('2023-01-01', periods=50, freq='D'))
    
    def test_dataperiod_enum_performance(self):
        """测试DataPeriod枚举的性能"""
        # 测试枚举转换的性能
        start_time = time.time()
        
        # 大量字符串到枚举的转换
        test_strings = ['1d', '1w', '1mo', '3mo', '6mo', '1y', '2y'] * 100
        
        for string_period in test_strings:
            period_enum = DataPeriod.from_string(string_period)
            assert isinstance(period_enum, DataPeriod)
        
        conversion_time = time.time() - start_time
        
        # 转换应该很快（不超过0.1秒）
        assert conversion_time < 0.1, f"DataPeriod转换太慢: {conversion_time:.3f}秒"
    
    def test_smart_caching_behavior(self, mock_data):
        """测试智能缓存行为"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟依赖项
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.records_count = 50
            mock_validation_result.issues = []
            
            # 第一次调用 - 缓存未命中，应该调用数据获取
            data_manager.cache_manager.get.return_value = None
            data_manager.cache_manager.set.return_value = None
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = mock_data
                        
                        start_time = time.time()
                        result1 = data_manager.get_stock_data('AAPL', period='1y')
                        first_call_time = time.time() - start_time
                        
                        # 验证第一次调用
                        assert result1 is not None
                        assert mock_fetch.called
                        
                        # 第二次调用 - 缓存命中，应该更快
                        data_manager.cache_manager.get.return_value = mock_data
                        mock_fetch.reset_mock()
                        
                        start_time = time.time()
                        result2 = data_manager.get_stock_data('AAPL', period='1y')
                        second_call_time = time.time() - start_time
                        
                        # 验证第二次调用（缓存命中）
                        assert result2 is not None
                        # 第二次调用不应该触发数据获取
                        assert not mock_fetch.called
                        # 第二次调用应该更快（但由于是mock，时间差异可能不明显）
                        assert second_call_time <= first_call_time * 2  # 宽松的性能检查
    
    def test_batch_download_threshold_optimization(self):
        """测试批次下载阈值优化"""
        from src.data.managers.batch_downloader import BatchDownloader
        from src.utils.date_range_utils import DateRange
        
        batch_downloader = BatchDownloader()
        
        # 测试小时间范围不需要批次下载
        small_range = DateRange(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31)  # 30天
        )
        
        should_batch_small = batch_downloader.should_use_batch_download(small_range, '1d')
        assert not should_batch_small, "小时间范围不应该使用批次下载"
        
        # 测试大时间范围需要批次下载
        large_range = DateRange(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31)  # 4年
        )
        
        should_batch_large = batch_downloader.should_use_batch_download(large_range, '1d')
        assert should_batch_large, "大时间范围应该使用批次下载"
    
    def test_smart_interval_recommendation(self):
        """测试智能间隔推荐性能"""
        # 测试不同周期的推荐间隔
        test_cases = [
            (DataPeriod.DAYS_1, DataSource.YFINANCE, '1m'),
            (DataPeriod.WEEK_1, DataSource.YFINANCE, '5m'), 
            (DataPeriod.MONTH_1, DataSource.YFINANCE, '1h'),
            (DataPeriod.YEAR_1, DataSource.YFINANCE, '1d'),
            (DataPeriod.YEAR_5, DataSource.YFINANCE, '1d'),
        ]
        
        start_time = time.time()
        
        for period, source, expected_category in test_cases:
            recommended = period.get_recommended_interval(source)
            # 验证推荐结果是合理的
            assert recommended is not None
            assert isinstance(recommended, str)
        
        recommendation_time = time.time() - start_time
        
        # 推荐应该很快
        assert recommendation_time < 0.01, f"间隔推荐太慢: {recommendation_time:.3f}秒"
    
    def test_smart_time_advisor_performance(self):
        """测试智能时间建议器的性能"""
        advisor = get_smart_time_advisor()
        
        start_time = time.time()
        
        # 测试多个建议请求
        use_cases = ['backtesting', 'training', 'validation', 'real_time']
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        for use_case in use_cases:
            for symbol in symbols:
                suggestion = advisor.suggest_optimal_time_range(
                    symbol=symbol,
                    use_case=use_case
                )
                
                # 验证建议结果
                assert suggestion is not None
                assert suggestion.data_source is not None
                assert suggestion.confidence > 0
        
        total_time = time.time() - start_time
        avg_time_per_suggestion = total_time / (len(use_cases) * len(symbols))
        
        # 每个建议应该很快生成
        assert avg_time_per_suggestion < 0.01, f"时间建议生成太慢: {avg_time_per_suggestion:.3f}秒"
    
    def test_date_range_calculation_performance(self, mock_data):
        """测试日期范围计算性能"""
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            start_time = time.time()
            
            # 测试多个周期到日期范围的转换
            periods = [
                DataPeriod.DAYS_7, DataPeriod.DAYS_30, DataPeriod.MONTH_1,
                DataPeriod.MONTH_3, DataPeriod.MONTH_6, DataPeriod.YEAR_1,
                DataPeriod.YEAR_2, DataPeriod.YEAR_5
            ]
            
            for period in periods:
                date_range = data_manager.convert_period_to_date_range(period)
                
                # 验证结果
                assert date_range is not None
                assert date_range.start_date is not None
                assert date_range.end_date is not None
                assert date_range.duration_days > 0
            
            conversion_time = time.time() - start_time
            avg_conversion_time = conversion_time / len(periods)
            
            # 每个转换应该很快
            assert avg_conversion_time < 0.001, f"日期范围转换太慢: {avg_conversion_time:.4f}秒"
    
    def test_memory_usage_optimization(self, mock_data):
        """测试内存使用优化"""
        # 这是一个概念测试，实际内存测试需要更复杂的工具
        config = Config()
        
        with patch('src.data.core.data_manager.get_routing_manager'), \
             patch('src.data.core.data_manager.get_compatibility_checker'), \
             patch('src.data.core.data_manager.get_cache_manager'), \
             patch('src.data.core.data_manager.MarketTypeDetector'):
            
            data_manager = DataManager(config)
            
            # 模拟依赖项
            data_manager.market_detector.detect.return_value = Mock(value='stock')
            data_manager.cache_manager.get.return_value = None
            
            # 创建大数据集来模拟内存使用
            large_mock_data = pd.DataFrame({
                'Open': [100.0] * 10000,
                'High': [105.0] * 10000,
                'Low': [99.0] * 10000,
                'Close': [104.0] * 10000,
                'Volume': [1000] * 10000
            }, index=pd.date_range('2020-01-01', periods=10000, freq='D'))
            
            mock_validation_result = Mock()
            mock_validation_result.is_valid = True
            mock_validation_result.records_count = 10000
            mock_validation_result.issues = []
            
            with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = large_mock_data
                
                with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                    mock_validate.return_value = mock_validation_result
                    
                    with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                        mock_clean.return_value = large_mock_data
                        
                        # 测试大数据处理
                        result = data_manager.get_stock_data('AAPL', period='max')
                        
                        # 验证结果
                        assert result is not None
                        assert len(result) == 10000
                        
                        # 验证数据类型优化（应该使用合适的数据类型）
                        assert result['Volume'].dtype in ['int64', 'int32', 'float64', 'float32']
    
    def test_concurrent_request_handling(self, mock_data):
        """测试并发请求处理"""
        import threading
        
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
            mock_validation_result.records_count = 50
            mock_validation_result.issues = []
            
            results = []
            errors = []
            
            def fetch_data_thread(symbol):
                try:
                    with patch.object(data_manager.data_fetcher, 'fetch_data') as mock_fetch:
                        mock_fetch.return_value = mock_data
                        
                        with patch.object(data_manager.data_validator, 'validate_data') as mock_validate:
                            mock_validate.return_value = mock_validation_result
                            
                            with patch.object(data_manager.data_validator, 'clean_data') as mock_clean:
                                mock_clean.return_value = mock_data
                                
                                result = data_manager.get_stock_data(symbol, period='1y')
                                results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # 创建多个线程并发请求
            threads = []
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            
            start_time = time.time()
            
            for symbol in symbols:
                thread = threading.Thread(target=fetch_data_thread, args=(symbol,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # 验证结果
            assert len(errors) == 0, f"并发请求出现错误: {errors}"
            assert len(results) == len(symbols), "并发请求结果数量不匹配"
            
            # 并发处理应该相对高效
            assert total_time < 5.0, f"并发处理太慢: {total_time:.3f}秒"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])