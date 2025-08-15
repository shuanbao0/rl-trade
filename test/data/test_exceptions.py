#!/usr/bin/env python3
"""
测试数据异常类
"""

import unittest
from datetime import datetime
from src.data.exceptions import (
    DataSourceError,
    DataSourceConnectionError,
    DataValidationError,
    DataInsufficientError,
    DataSourceCompatibilityError,
    AllDataSourcesFailedError
)


class TestDataExceptions(unittest.TestCase):
    """数据异常类测试"""
    
    def test_data_source_error(self):
        """测试基础数据源错误"""
        error = DataSourceError(
            "Test error",
            source="yfinance",
            symbol="AAPL",
            details={"code": 404}
        )
        
        self.assertEqual(error.source, "yfinance")
        self.assertEqual(error.symbol, "AAPL")
        self.assertEqual(error.details, {"code": 404})
        self.assertIn("yfinance", str(error))
        self.assertIn("AAPL", str(error))
    
    def test_connection_error(self):
        """测试连接错误"""
        error = DataSourceConnectionError(
            "Connection failed",
            source="truefx",
            symbol="EURUSD"
        )
        
        self.assertIn("Connection failed", str(error))
    
    def test_validation_error(self):
        """测试数据验证错误"""
        error = DataValidationError(
            "Invalid data format",
            source="yfinance",
            symbol="AAPL",
            validation_issues=["missing_volume", "negative_prices"]
        )
        
        self.assertEqual(error.validation_issues, ["missing_volume", "negative_prices"])
        self.assertIn("Invalid data format", str(error))
    
    def test_insufficient_data_error(self):
        """测试数据不足错误"""
        error = DataInsufficientError(
            "Not enough data",
            source="oanda",
            symbol="GBPUSD",
            expected_records=1000,
            actual_records=50
        )
        
        self.assertEqual(error.expected_records, 1000)
        self.assertEqual(error.actual_records, 50)
        self.assertIn("Not enough data", str(error))
    
    def test_compatibility_error(self):
        """测试兼容性错误"""
        error = DataSourceCompatibilityError(
            "Source not compatible",
            source="truefx",
            symbol="AAPL",
            compatibility_score=0.1,
            requirements={"market_type": "stock", "interval": "1d"}
        )
        
        self.assertEqual(error.requirements["market_type"], "stock")
        self.assertEqual(error.requirements["interval"], "1d")
        self.assertEqual(error.compatibility_score, 0.1)
    
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
        self.assertEqual(error.source_errors, source_errors)
        self.assertEqual(error.symbol, "EURUSD")
        
        # 测试字符串表示包含所有信息
        error_str = str(error)
        self.assertIn("All sources failed", error_str)
        self.assertIn("EURUSD", error_str)
        # 验证failed_sources和source_errors属性存在
        self.assertEqual(len(error.failed_sources), 3)
        self.assertEqual(len(error.source_errors), 3)
    
    def test_error_inheritance(self):
        """测试错误继承关系"""
        # 所有错误都应该继承自DataSourceError
        connection_error = DataSourceConnectionError("test", source="test", symbol="TEST")
        validation_error = DataValidationError("test", source="test", symbol="TEST")
        insufficient_error = DataInsufficientError("test", source="test", symbol="TEST")
        compatibility_error = DataSourceCompatibilityError("test", source="test", symbol="TEST")
        all_failed_error = AllDataSourcesFailedError("test", failed_sources=[], symbol="TEST")
        
        self.assertIsInstance(connection_error, DataSourceError)
        self.assertIsInstance(validation_error, DataSourceError)
        self.assertIsInstance(insufficient_error, DataSourceError)
        self.assertIsInstance(compatibility_error, DataSourceError)
        self.assertIsInstance(all_failed_error, DataSourceError)
    
    def test_error_with_none_values(self):
        """测试错误处理None值"""
        error = DataSourceError(
            "Test error",
            source=None,
            symbol=None,
            details=None
        )
        
        # 确保不会因为None值而崩溃
        error_str = str(error)
        self.assertIsInstance(error_str, str)
        self.assertIn("Test error", error_str)


if __name__ == '__main__':
    unittest.main()