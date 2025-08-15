#!/usr/bin/env python3
"""
测试分批次下载器与数据模块的集成
BatchDownloader功能已集成到DataManager中
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta

from src.data import DataManager
from src.data.sources.base import MarketType
from src.utils.config import Config


class TestBatchDownloaderIntegration(unittest.TestCase):
    """分批次下载器集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Open': [100.0 + i * 0.1 for i in range(100)],
            'High': [102.0 + i * 0.1 for i in range(100)],
            'Low': [99.0 + i * 0.1 for i in range(100)],
            'Close': [101.0 + i * 0.1 for i in range(100)],
            'Volume': [1000 + i * 10 for i in range(100)]
        }, index=dates)
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_data_manager_batch_integration(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试DataManager的分批次下载功能集成"""
        
        # 设置mocks
        mock_detector.return_value.detect_market_type.return_value = MarketType.STOCK
        mock_cache.return_value.get.return_value = None
        mock_cache.return_value.put.return_value = True
        
        # 创建DataManager
        data_manager = DataManager(self.config)
        
        # 验证分批次下载相关属性已正确初始化
        self.assertIsNotNone(data_manager.batch_cache_dir)
        self.assertIsInstance(data_manager.batch_config, dict)
        self.assertIn('auto_enable_threshold_days', data_manager.batch_config)
        self.assertIn('default_batch_days', data_manager.batch_config)
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')  
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_smart_batch_detection(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试智能分批次下载检测"""
        
        # 设置mocks
        mock_detector.return_value.detect_market_type.return_value = MarketType.STOCK
        mock_cache.return_value.get.return_value = None
        
        data_manager = DataManager(self.config)
        
        # 测试不同场景的分批次检测
        test_cases = [
            # (period, interval, expected_batch)
            ('1y', '1d', False),      # 普通情况，不需要分批次
            ('5y', '1d', True),       # 长时间跨度，需要分批次  
            ('30d', '1m', False),     # 高频但短时间，不需要分批次
            ('60d', '1m', True),      # 高频且长时间，需要分批次
            ('max', '1h', True),      # 最大时间范围，需要分批次
        ]
        
        for period, interval, expected in test_cases:
            should_batch = data_manager._should_use_batch_download(period, interval)
            self.assertEqual(should_batch, expected, 
                           f"Period: {period}, Interval: {interval} should {'use' if expected else 'not use'} batch download")
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_period_parsing(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试周期解析功能"""
        
        data_manager = DataManager(self.config)
        
        test_cases = [
            ('1d', 1),
            ('7d', 7),
            ('30d', 30),
            ('1w', 7),
            ('4w', 28),
            ('1mo', 30),
            ('6mo', 180),
            ('1y', 365),
            ('2y', 730),
            ('max', 7300),  # 20年
        ]
        
        for period_str, expected_days in test_cases:
            parsed_days = data_manager._parse_period_to_days(period_str)
            self.assertEqual(parsed_days, expected_days, f"Period {period_str} should parse to {expected_days} days")
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_record_estimation(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试记录数量估算"""
        
        data_manager = DataManager(self.config)
        
        test_cases = [
            # (days, interval, expected_records)
            (1, '1m', 1440),      # 1天的1分钟数据
            (1, '1h', 24),        # 1天的1小时数据
            (1, '1d', 1),         # 1天的日线数据
            (30, '1m', 43200),    # 30天的1分钟数据
            (365, '1d', 365),     # 1年的日线数据
        ]
        
        for days, interval, expected in test_cases:
            estimated = data_manager._estimate_record_count(days, interval)
            self.assertEqual(estimated, expected, 
                           f"Days: {days}, Interval: {interval} should estimate {expected} records")
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    @patch('src.data.data_manager.DataManager._fetch_from_data_source')
    def test_smart_fetch_routing(self, mock_fetch, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试智能数据获取路由"""
        
        # 设置mocks
        mock_detector.return_value.detect_market_type.return_value = MarketType.STOCK
        mock_cache.return_value.get.return_value = None
        mock_fetch.return_value = self.test_data
        
        data_manager = DataManager(self.config)
        
        # Mock批次下载器
        with patch.object(data_manager, '_fetch_with_batch_download') as mock_batch_fetch:
            mock_batch_fetch.return_value = self.test_data
            
            # 测试常规下载路由
            result1 = data_manager._smart_fetch_data('AAPL', '1y', '1d')
            mock_fetch.assert_called_once()
            mock_batch_fetch.assert_not_called()
            
            # 重置mocks
            mock_fetch.reset_mock()
            mock_batch_fetch.reset_mock()
            
            # 测试分批次下载路由  
            result2 = data_manager._smart_fetch_data('AAPL', '5y', '1m')
            mock_fetch.assert_not_called()
            mock_batch_fetch.assert_called_once()
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_batch_download_estimation(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试分批次下载时间估算"""
        
        data_manager = DataManager(self.config)
        
        # Mock DataManager的内置估算方法
        with patch.object(data_manager, '_estimate_download_time') as mock_estimate:
            mock_estimate.return_value = {
                'total_days': 1825,
                'batch_days': 30,
                'total_batches': 61,
                'estimated_time_per_batch_seconds': 10,
                'total_estimated_time_seconds': 610,
                'total_estimated_time_minutes': 10.17,
                'total_estimated_time_hours': 0.17
            }
            
            estimation = data_manager.get_batch_download_estimation('AAPL', '5y', '1d')
            
            # 验证估算结果
            self.assertIsInstance(estimation, dict)
            self.assertIn('total_days', estimation)
            self.assertIn('total_batches', estimation)
            self.assertIn('total_estimated_time_minutes', estimation)
            
            # 验证调用了正确的方法
            mock_estimate.assert_called_once()
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_batch_methods_integrated(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试分批次下载方法已集成到DataManager"""
        
        data_manager = DataManager(self.config)
        
        # 验证分批次下载方法存在
        self.assertTrue(hasattr(data_manager, '_download_in_batches'))
        self.assertTrue(hasattr(data_manager, '_calculate_optimal_batch_size'))
        self.assertTrue(hasattr(data_manager, '_generate_batches'))
        self.assertTrue(hasattr(data_manager, '_estimate_download_time'))
        
        # 测试估算功能
        estimation = data_manager._estimate_download_time(
            'AAPL', '2023-01-01', '2023-12-31', '1d'
        )
        
        self.assertIsInstance(estimation, dict)
        self.assertIn('total_days', estimation)
        self.assertEqual(estimation['total_days'], 364)
    
    @patch('src.data.data_manager.get_routing_manager')
    @patch('src.data.data_manager.get_compatibility_checker')
    @patch('src.data.data_manager.get_cache_manager')
    @patch('src.data.data_manager.MarketTypeDetector')
    def test_circular_dependency_resolved(self, mock_detector, mock_cache, mock_compat, mock_routing):
        """测试循环依赖已解决"""
        
        # 创建DataManager不应该引发循环依赖错误
        try:
            data_manager = DataManager(self.config)
            
            # 验证分批次下载功能可以直接使用，不需要外部BatchDownloader
            self.assertIsNotNone(data_manager.batch_cache_dir)
            self.assertTrue(data_manager.batch_cache_dir.exists())
            
            # 测试批次大小计算
            batch_size = data_manager._calculate_optimal_batch_size('1d', 365)
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)
            
        except ImportError as e:
            self.fail(f"循环依赖错误: {e}")


if __name__ == '__main__':
    unittest.main()