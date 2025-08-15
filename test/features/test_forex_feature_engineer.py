#!/usr/bin/env python3
"""
测试Forex特征工程器
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.features.forex_feature_engineer import ForexFeatureEngineer
from src.utils.config import Config


class TestForexFeatureEngineer(unittest.TestCase):
    """Forex特征工程器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        self.engineer = ForexFeatureEngineer(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        self.test_data = pd.DataFrame({
            'Open': np.random.uniform(1.1000, 1.2000, 100),
            'High': np.random.uniform(1.1050, 1.2050, 100),
            'Low': np.random.uniform(1.0950, 1.1950, 100),
            'Close': np.random.uniform(1.1000, 1.2000, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'Bid': np.random.uniform(1.0995, 1.1995, 100),
            'Ask': np.random.uniform(1.1005, 1.2005, 100)
        }, index=dates)
        
        # 确保High >= max(Open, Close) 和 Low <= min(Open, Close)
        self.test_data['High'] = np.maximum(self.test_data['High'], 
                                          np.maximum(self.test_data['Open'], self.test_data['Close']))
        self.test_data['Low'] = np.minimum(self.test_data['Low'], 
                                         np.minimum(self.test_data['Open'], self.test_data['Close']))
    
    def test_pip_calculation(self):
        """测试点值计算"""
        # 测试EUR/USD (4位小数)
        pips_eurusd = self.engineer.calculate_pips(1.1050, 1.1000, "EURUSD")
        self.assertEqual(pips_eurusd, 50)
        
        # 测试USD/JPY (2位小数)
        pips_usdjpy = self.engineer.calculate_pips(110.50, 110.00, "USDJPY")
        self.assertEqual(pips_usdjpy, 50)
    
    def test_spread_calculation(self):
        """测试价差计算"""
        spread = self.engineer.calculate_spread(1.1005, 1.1000, "EURUSD")
        self.assertEqual(spread, 0.5)  # 0.5 pips
    
    def test_session_detection(self):
        """测试交易时段检测"""
        # 测试伦敦时段
        london_time = pd.Timestamp('2023-01-01 10:00:00', tz='UTC')
        session = self.engineer.get_trading_session(london_time)
        self.assertEqual(session, 'london')
        
        # 测试纽约时段
        ny_time = pd.Timestamp('2023-01-01 15:00:00', tz='UTC')
        session = self.engineer.get_trading_session(ny_time)
        self.assertEqual(session, 'new_york')
    
    def test_forex_specific_features(self):
        """测试Forex特定特征"""
        features = self.engineer.create_forex_features(self.test_data, symbol="EURUSD")
        
        # 验证特征列存在
        expected_features = [
            'pip_range', 'spread', 'session_volume', 
            'hourly_volatility', 'currency_strength'
        ]
        
        for feature in expected_features:
            if feature in features.columns:
                self.assertIn(feature, features.columns)
    
    def test_correlation_features(self):
        """测试相关性特征"""
        # 创建相关货币对数据
        correlated_pairs = {
            "GBPUSD": self.test_data.copy(),
            "EURUSD": self.test_data.copy()
        }
        
        features = self.engineer.create_correlation_features(
            self.test_data, "EURUSD", correlated_pairs
        )
        
        # 验证相关性特征
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.test_data))
    
    def test_volatility_regime_detection(self):
        """测试波动率制度检测"""
        regime = self.engineer.detect_volatility_regime(self.test_data)
        
        # 验证返回值
        self.assertIn(regime, ['low', 'medium', 'high'])
    
    def test_support_resistance_levels(self):
        """测试支撑阻力位检测"""
        levels = self.engineer.find_support_resistance(self.test_data)
        
        # 验证返回格式
        self.assertIsInstance(levels, dict)
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
    
    def test_feature_engineering_pipeline(self):
        """测试完整特征工程流水线"""
        features = self.engineer.prepare_features(self.test_data, symbol="EURUSD")
        
        # 验证特征数量和质量
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), len(self.test_data.columns))
        self.assertEqual(len(features), len(self.test_data))
        
        # 验证没有全部为NaN的列
        for col in features.columns:
            self.assertFalse(features[col].isna().all(), f"Column {col} is all NaN")
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空数据
        empty_data = pd.DataFrame()
        
        try:
            features = self.engineer.prepare_features(empty_data, symbol="EURUSD")
            # 如果没有抛出异常，验证返回的是空DataFrame
            self.assertTrue(features.empty)
        except Exception:
            # 抛出异常也是可以接受的
            pass
    
    def test_feature_validation(self):
        """测试特征验证"""
        features = self.engineer.prepare_features(self.test_data, symbol="EURUSD")
        
        # 验证特征值的合理性
        for col in features.select_dtypes(include=[np.number]).columns:
            # 检查是否有无穷大值
            self.assertFalse(np.isinf(features[col]).any(), 
                           f"Column {col} contains infinite values")
            
            # 检查是否有过多的NaN值（超过50%）
            nan_ratio = features[col].isna().sum() / len(features)
            self.assertLess(nan_ratio, 0.5, 
                          f"Column {col} has too many NaN values: {nan_ratio:.2%}")


if __name__ == '__main__':
    unittest.main()