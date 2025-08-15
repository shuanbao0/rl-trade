#!/usr/bin/env python3
"""
测试数据管理模块的错误处理功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.data import (
    DataManager,
    DataSourceError,
    DataSourceConnectionError,
    DataValidationError,
    DataInsufficientError,
    DataSourceCompatibilityError,
    AllDataSourcesFailedError
)
from src.data.sources.base import MarketType
from src.utils.config import Config


class TestErrorHandling(unittest.TestCase):
    """错误处理测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        # 创建模拟的依赖项
        with patch('src.data.data_manager.get_routing_manager'), \
             patch('src.data.data_manager.get_compatibility_checker'), \
             patch('src.data.data_manager.get_cache_manager'), \
             patch('src.data.data_manager.MarketTypeDetector'):
            self.data_manager = DataManager(self.config)
    
    def test_data_source_error_creation(self):
        """测试数据源错误创建"""
        error = DataSourceError(
            "Test error message",
            source="yfinance",
            symbol="AAPL",
            details={"code": 404}
        )
        
        self.assertEqual(str(error), "Test error message | Source: yfinance | Symbol: AAPL | Details: code=404")
        self.assertEqual(error.source, "yfinance")
        self.assertEqual(error.symbol, "AAPL")
        self.assertEqual(error.details["code"], 404)
    
    def test_data_insufficient_error(self):
        """测试数据不足错误"""
        error = DataInsufficientError(
            "Not enough data",
            expected_records=100,
            actual_records=50,
            symbol="AAPL"
        )
        
        self.assertEqual(error.expected_records, 100)
        self.assertEqual(error.actual_records, 50)
        self.assertEqual(error.symbol, "AAPL")
    
    def test_compatibility_error(self):
        """测试兼容性错误"""
        error = DataSourceCompatibilityError(
            "Low compatibility",
            compatibility_score=0.3,
            requirements={"market_type": "forex", "min_score": 0.7},
            source="yfinance"
        )
        
        self.assertEqual(error.compatibility_score, 0.3)
        self.assertEqual(error.requirements["market_type"], "forex")
        self.assertEqual(error.source, "yfinance")
    
    def test_all_sources_failed_error(self):
        """测试所有数据源失败错误"""
        failed_sources = ["yfinance", "truefx", "oanda"]
        source_errors = {
            "yfinance": "API limit exceeded",
            "truefx": "Connection timeout",
            "oanda": "Authentication failed"
        }
        
        error = AllDataSourcesFailedError(
            "All sources failed",
            failed_sources=failed_sources,
            source_errors=source_errors,
            symbol="EURUSD"
        )
        
        self.assertEqual(error.failed_sources, failed_sources)
        self.assertEqual(error.source_errors["yfinance"], "API limit exceeded")
        self.assertEqual(error.symbol, "EURUSD")
    
    @patch('src.data.sources.DataSourceRegistry')
    def test_unregistered_source_error(self, mock_registry):
        """测试未注册数据源错误"""
        # 模拟未注册的数据源
        mock_registry.is_registered.return_value = False
        mock_registry.list_sources.return_value = ["yfinance", "truefx"]
        
        with self.assertRaises(DataSourceError) as cm:
            self.data_manager._fetch_from_specified_source(
                source="unknown_source",
                symbol="AAPL",
                market_type=MarketType.STOCK,
                period="1y",
                interval="1d"
            )
        
        self.assertIn("not registered", str(cm.exception))
        self.assertIn("unknown_source", str(cm.exception))
    
    @patch('src.data.sources.DataSourceRegistry')
    def test_compatibility_check_failure(self, mock_registry):
        """测试兼容性检查失败"""
        # 模拟已注册的数据源
        mock_registry.is_registered.return_value = True
        
        # 模拟兼容性检查器返回低分
        mock_result = Mock()
        mock_result.overall_score = 0.3  # 低于0.7的阈值
        self.data_manager.compatibility_checker.check_compatibility.return_value = mock_result
        
        with self.assertRaises(DataSourceCompatibilityError) as cm:
            self.data_manager._fetch_from_specified_source(
                source="yfinance",
                symbol="EURUSD",
                market_type=MarketType.FOREX,
                period="1y",
                interval="1m"
            )
        
        self.assertIn("low compatibility", str(cm.exception))
        self.assertEqual(cm.exception.compatibility_score, 0.3)
    
    def test_data_validation_error_insufficient_records(self):
        """测试数据验证错误 - 记录不足"""
        # 创建不足的测试数据
        insufficient_data = pd.DataFrame({
            'Close': [100, 101]  # 只有2条记录
        })
        
        # 模拟验证结果
        with patch.object(self.data_manager, 'validate_data') as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=False,
                records_count=2,
                issues=["Insufficient records"]
            )
            
            with self.assertRaises(DataInsufficientError) as cm:
                self.data_manager._process_market_data(
                    insufficient_data, 
                    MarketType.STOCK, 
                    "AAPL"
                )
            
            self.assertEqual(cm.exception.actual_records, 2)
            self.assertEqual(cm.exception.symbol, "AAPL")
    
    @patch('src.data.data_manager.DataSourceFactory')
    def test_specified_source_failure_no_fallback(self, mock_factory):
        """测试指定数据源失败时不使用fallback"""
        # 模拟数据源创建失败
        mock_factory.create_data_source.side_effect = Exception("Connection failed")
        
        with patch('src.data.sources.DataSourceRegistry') as mock_registry:
            mock_registry.is_registered.return_value = True
            
            # 跳过兼容性检查
            self.data_manager.compatibility_checker.check_compatibility.return_value = Mock(overall_score=0.9)
            
            with self.assertRaises(DataSourceError) as cm:
                self.data_manager._fetch_from_specified_source(
                    source="yfinance",
                    symbol="AAPL",
                    market_type=MarketType.STOCK,
                    period="1y",
                    interval="1d"
                )
            
            # 验证错误消息包含指定数据源信息
            self.assertIn("yfinance", str(cm.exception))
            self.assertIn("failed", str(cm.exception))
            self.assertEqual(cm.exception.source, "yfinance")
            self.assertEqual(cm.exception.symbol, "AAPL")
    
    def test_auto_routing_fallback_success(self):
        """测试自动路由时的fallback成功"""
        # 模拟路由管理器返回主源和备选源
        mock_routing_manager = self.data_manager.routing_manager
        mock_routing_manager.get_optimal_sources.return_value = ["yfinance", "truefx"]
        
        # 模拟主数据源失败，备选数据源成功
        mock_data = pd.DataFrame({
            'Open': [100], 'High': [102], 'Low': [99], 'Close': [101], 'Volume': [1000]
        })
        
        with patch.object(self.data_manager, '_fetch_from_single_source') as mock_fetch:
            # 第一次调用（主源）失败，第二次调用（备选源）成功
            mock_fetch.side_effect = [Exception("Primary failed"), mock_data]
            
            with patch.object(self.data_manager, '_process_market_data') as mock_process:
                mock_process.return_value = mock_data
                
                # 模拟缓存管理器
                self.data_manager.cache_manager.get.return_value = None
                
                result = self.data_manager._fetch_with_routing(
                    symbol="AAPL",
                    market_type=MarketType.STOCK,
                    period="1y",
                    interval="1d"
                )
                
                # 验证成功获取数据
                self.assertIsNotNone(result)
                self.assertEqual(len(result), 1)
                
                # 验证调用了两次fetch（主源和备选源）
                self.assertEqual(mock_fetch.call_count, 2)
    
    def test_auto_routing_all_sources_fail(self):
        """测试自动路由时所有数据源都失败"""
        # 模拟路由管理器返回多个数据源
        mock_routing_manager = self.data_manager.routing_manager
        mock_routing_manager.get_optimal_sources.return_value = ["yfinance", "truefx", "oanda"]
        
        with patch.object(self.data_manager, '_fetch_from_single_source') as mock_fetch:
            # 所有数据源都失败
            mock_fetch.side_effect = Exception("All sources failed")
            
            # 模拟缓存管理器
            self.data_manager.cache_manager.get.return_value = None
            
            with self.assertRaises(AllDataSourcesFailedError) as cm:
                self.data_manager._fetch_with_routing(
                    symbol="AAPL",
                    market_type=MarketType.STOCK,
                    period="1y",
                    interval="1d"
                )
            
            # 验证异常包含正确信息
            self.assertEqual(cm.exception.failed_sources, ["yfinance", "truefx", "oanda"])
            self.assertEqual(cm.exception.symbol, "AAPL")
            
            # 验证尝试了所有数据源
            self.assertEqual(mock_fetch.call_count, 3)
    
    def test_error_handling_maintains_context(self):
        """测试错误处理保持上下文信息"""
        error = DataSourceError(
            "Test error",
            source="yfinance",
            symbol="AAPL",
            details={"timestamp": "2023-01-01", "attempt": 3}
        )
        
        # 验证错误信息包含所有上下文
        error_str = str(error)
        self.assertIn("Test error", error_str)
        self.assertIn("yfinance", error_str)
        self.assertIn("AAPL", error_str)
        self.assertIn("timestamp=2023-01-01", error_str)
        self.assertIn("attempt=3", error_str)


if __name__ == '__main__':
    unittest.main()