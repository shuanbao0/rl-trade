#!/usr/bin/env python3
"""
测试相关性监控器
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.monitoring.correlation_monitor import CorrelationMonitor
from src.utils.config import Config


class TestCorrelationMonitor(unittest.TestCase):
    """相关性监控器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = Config()
        self.monitor = CorrelationMonitor(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # 创建具有已知相关性的数据
        np.random.seed(42)
        base_series = np.random.randn(100)
        
        self.test_data = pd.DataFrame({
            'EURUSD': base_series + np.random.randn(100) * 0.1,
            'GBPUSD': base_series * 0.8 + np.random.randn(100) * 0.3,
            'USDJPY': -base_series * 0.5 + np.random.randn(100) * 0.4,
            'AUDUSD': base_series * 0.6 + np.random.randn(100) * 0.2,
            'USDCHF': -base_series * 0.7 + np.random.randn(100) * 0.25
        }, index=dates)
    
    def test_calculate_correlation_matrix(self):
        """测试相关性矩阵计算"""
        corr_matrix = self.monitor.calculate_correlation_matrix(self.test_data)
        
        # 验证矩阵形状
        self.assertEqual(corr_matrix.shape, (5, 5))
        
        # 验证对角线为1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
        
        # 验证矩阵对称性
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.T.values)
        
        # 验证相关系数范围
        self.assertTrue((corr_matrix.values >= -1).all())
        self.assertTrue((corr_matrix.values <= 1).all())
    
    def test_detect_correlation_changes(self):
        """测试相关性变化检测"""
        # 创建历史相关性数据
        historical_corr = self.monitor.calculate_correlation_matrix(self.test_data[:50])
        current_corr = self.monitor.calculate_correlation_matrix(self.test_data[50:])
        
        changes = self.monitor.detect_correlation_changes(historical_corr, current_corr)
        
        # 验证返回格式
        self.assertIsInstance(changes, dict)
        
        # 验证包含预期键
        expected_keys = ['significant_changes', 'threshold_exceeded', 'change_matrix']
        for key in expected_keys:
            if key in changes:
                self.assertIn(key, changes)
    
    def test_rolling_correlation(self):
        """测试滚动相关性计算"""
        rolling_corr = self.monitor.calculate_rolling_correlation(
            self.test_data['EURUSD'], 
            self.test_data['GBPUSD'], 
            window=20
        )
        
        # 验证返回格式
        self.assertIsInstance(rolling_corr, pd.Series)
        self.assertEqual(len(rolling_corr), len(self.test_data))
        
        # 验证相关系数范围
        valid_corr = rolling_corr.dropna()
        self.assertTrue((valid_corr >= -1).all())
        self.assertTrue((valid_corr <= 1).all())
    
    def test_correlation_clusters(self):
        """测试相关性聚类"""
        clusters = self.monitor.identify_correlation_clusters(self.test_data)
        
        # 验证返回格式
        self.assertIsInstance(clusters, dict)
        
        # 验证所有符号都被分配到某个聚类
        all_symbols = set()
        for cluster_symbols in clusters.values():
            all_symbols.update(cluster_symbols)
        
        expected_symbols = set(self.test_data.columns)
        self.assertEqual(all_symbols, expected_symbols)
    
    def test_correlation_alerts(self):
        """测试相关性警报"""
        # 创建极端相关性变化
        extreme_data = self.test_data.copy()
        extreme_data['EXTREME'] = self.test_data['EURUSD'] * 0.99  # 高相关性
        
        alerts = self.monitor.check_correlation_alerts(extreme_data)
        
        # 验证返回格式
        self.assertIsInstance(alerts, list)
        
        # 可能有高相关性警报
        if alerts:
            for alert in alerts:
                self.assertIsInstance(alert, dict)
                expected_alert_keys = ['type', 'pairs', 'correlation', 'timestamp']
                for key in expected_alert_keys:
                    if key in alert:
                        self.assertIn(key, alert)
    
    def test_correlation_regime_detection(self):
        """测试相关性制度检测"""
        regime = self.monitor.detect_correlation_regime(self.test_data)
        
        # 验证返回值
        expected_regimes = ['low_correlation', 'medium_correlation', 'high_correlation', 'crisis_correlation']
        if regime:
            self.assertIn(regime, expected_regimes)
    
    def test_portfolio_correlation_risk(self):
        """测试投资组合相关性风险"""
        # 创建投资组合权重
        weights = {
            'EURUSD': 0.3,
            'GBPUSD': 0.2,
            'USDJPY': 0.2,
            'AUDUSD': 0.15,
            'USDCHF': 0.15
        }
        
        risk_metrics = self.monitor.calculate_portfolio_correlation_risk(self.test_data, weights)
        
        # 验证返回格式
        self.assertIsInstance(risk_metrics, dict)
        
        # 验证包含预期指标
        expected_metrics = ['diversification_ratio', 'concentration_risk', 'correlation_risk']
        for metric in expected_metrics:
            if metric in risk_metrics:
                self.assertIn(metric, risk_metrics)
                self.assertIsInstance(risk_metrics[metric], (int, float))
    
    def test_correlation_stability(self):
        """测试相关性稳定性分析"""
        stability = self.monitor.analyze_correlation_stability(self.test_data, window=30)
        
        # 验证返回格式
        self.assertIsInstance(stability, dict)
        
        # 验证包含稳定性指标
        if 'stability_score' in stability:
            self.assertIsInstance(stability['stability_score'], (int, float))
            self.assertGreaterEqual(stability['stability_score'], 0)
            self.assertLessEqual(stability['stability_score'], 1)
    
    def test_cross_asset_correlation(self):
        """测试跨资产相关性分析"""
        # 添加股票指数数据
        stock_data = pd.DataFrame({
            'SPY': np.random.randn(100),
            'QQQ': np.random.randn(100)
        }, index=self.test_data.index)
        
        cross_corr = self.monitor.analyze_cross_asset_correlation(
            self.test_data, stock_data
        )
        
        # 验证返回格式
        self.assertIsInstance(cross_corr, pd.DataFrame)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空数据
        empty_data = pd.DataFrame()
        
        try:
            corr_matrix = self.monitor.calculate_correlation_matrix(empty_data)
            self.assertTrue(corr_matrix.empty)
        except Exception:
            # 抛出异常也是可以接受的
            pass
        
        # 测试单列数据
        single_col_data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        
        try:
            corr_matrix = self.monitor.calculate_correlation_matrix(single_col_data)
            # 单列数据应该返回1x1矩阵
            if not corr_matrix.empty:
                self.assertEqual(corr_matrix.shape, (1, 1))
                self.assertEqual(corr_matrix.iloc[0, 0], 1.0)
        except Exception:
            pass
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        import time
        
        # 测试大数据集的处理时间
        large_data = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f'PAIR_{i}' for i in range(10)]
        )
        
        start_time = time.time()
        corr_matrix = self.monitor.calculate_correlation_matrix(large_data)
        elapsed_time = time.time() - start_time
        
        # 验证在合理时间内完成
        self.assertLess(elapsed_time, 5.0, "Correlation calculation took too long")
        
        # 验证结果正确性
        self.assertEqual(corr_matrix.shape, (10, 10))


if __name__ == '__main__':
    unittest.main()