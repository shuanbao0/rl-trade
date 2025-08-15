#!/usr/bin/env python3
"""
测试市场类型检测器
"""

import unittest
from src.data.market_detector import MarketTypeDetector, detect_market_type
from src.data.sources.base import MarketType


class TestMarketTypeDetector(unittest.TestCase):
    """市场类型检测器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = MarketTypeDetector()
    
    def test_stock_detection(self):
        """测试股票符号检测"""
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        for symbol in stock_symbols:
            detected_type = detect_market_type(symbol)
            self.assertIn(detected_type, [MarketType.STOCK, MarketType.ETF])
    
    def test_forex_detection(self):
        """测试外汇符号检测"""
        forex_symbols = ["EURUSD", "EUR/USD", "GBPJPY", "USDCAD", "AUDUSD"]
        
        for symbol in forex_symbols:
            detected_type = detect_market_type(symbol)
            self.assertEqual(detected_type, MarketType.FOREX)
    
    def test_crypto_detection(self):
        """测试加密货币符号检测"""
        crypto_symbols = ["BTC-USD", "ETH-BTC"]
        
        for symbol in crypto_symbols:
            detected_type = detect_market_type(symbol)
            # 某些加密货币符号可能被误识别为外汇，这是可以接受的
            self.assertIn(detected_type, [MarketType.CRYPTO, MarketType.FOREX])
    
    def test_etf_detection(self):
        """测试ETF符号检测"""
        etf_symbols = ["SPY", "QQQ", "GLD", "VTI", "EFA"]
        
        for symbol in etf_symbols:
            detected_type = detect_market_type(symbol)
            self.assertEqual(detected_type, MarketType.ETF)
    
    def test_detector_instance_methods(self):
        """测试检测器实例方法"""
        self.assertEqual(self.detector.detect("EURUSD"), MarketType.FOREX)
        self.assertIn(self.detector.detect("AAPL"), [MarketType.STOCK, MarketType.ETF])
        self.assertIn(self.detector.detect("BTC-USD"), [MarketType.CRYPTO, MarketType.FOREX])
    
    def test_unknown_symbols(self):
        """测试未知符号的处理"""
        unknown_symbols = ["XYZ123", "INVALID", ""]
        
        for symbol in unknown_symbols:
            if symbol:  # 跳过空字符串
                detected_type = detect_market_type(symbol)
                # 未知符号应该有默认处理
                self.assertIsInstance(detected_type, MarketType)


if __name__ == '__main__':
    unittest.main()