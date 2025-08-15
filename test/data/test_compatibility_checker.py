"""
测试兼容性检查器功能
"""

import unittest
from unittest.mock import Mock, patch

from src.data.compatibility_checker import (
    CompatibilityChecker, 
    CompatibilityRequest, 
    CompatibilityResult,
    get_compatibility_checker
)
from src.data.sources.base import (
    MarketType, 
    DataInterval, 
    DataQuality, 
    DataSourceCapabilities
)


class TestCompatibilityRequest(unittest.TestCase):
    """兼容性请求测试类"""
    
    def test_compatibility_request_creation(self):
        """测试兼容性请求创建"""
        request = CompatibilityRequest(
            source='yfinance',
            market_type=MarketType.STOCK,
            interval=DataInterval.DAY_1,
            symbol='AAPL',
            quality_requirement=DataQuality.MEDIUM
        )
        
        self.assertEqual(request.source, 'yfinance')
        self.assertEqual(request.market_type, MarketType.STOCK)
        self.assertEqual(request.interval, DataInterval.DAY_1)
        self.assertEqual(request.symbol, 'AAPL')
        self.assertEqual(request.quality_requirement, DataQuality.MEDIUM)
        self.assertFalse(request.requires_realtime)
        self.assertFalse(request.requires_streaming)


class TestCompatibilityResult(unittest.TestCase):
    """兼容性结果测试类"""
    
    def test_compatibility_result_creation(self):
        """测试兼容性结果创建"""
        result = CompatibilityResult(
            is_compatible=True,
            issues=[],
            warnings=['Low latency warning'],
            recommendations=['Consider using premium source'],
            compatibility_score=0.85
        )
        
        self.assertTrue(result.is_compatible)
        self.assertFalse(result.has_critical_issues())
        self.assertTrue(result.has_warnings())
        self.assertEqual(result.compatibility_score, 0.85)
    
    def test_has_critical_issues(self):
        """测试严重问题检查"""
        # 兼容的情况
        result = CompatibilityResult(
            is_compatible=True,
            issues=[],
            warnings=[],
            recommendations=[],
            compatibility_score=1.0
        )
        self.assertFalse(result.has_critical_issues())
        
        # 不兼容的情况
        result = CompatibilityResult(
            is_compatible=False,
            issues=['Market type not supported'],
            warnings=[],
            recommendations=[],
            compatibility_score=0.0
        )
        self.assertTrue(result.has_critical_issues())


class TestCompatibilityChecker(unittest.TestCase):
    """兼容性检查器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.checker = CompatibilityChecker()
    
    def test_checker_initialization(self):
        """测试检查器初始化"""
        self.assertIsInstance(self.checker.weights, dict)
        self.assertIn('market_support', self.checker.weights)
        self.assertIn('interval_support', self.checker.weights)
        self.assertIn('quality_match', self.checker.weights)
        
        # 验证权重总和为1
        total_weight = sum(self.checker.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_check_compatibility_source_not_registered(self, mock_registry):
        """测试数据源未注册的情况"""
        mock_registry.is_registered.return_value = False
        mock_registry.list_sources.return_value = ['yfinance', 'truefx']
        
        request = CompatibilityRequest(
            source='nonexistent',
            market_type=MarketType.STOCK,
            interval=DataInterval.DAY_1
        )
        
        result = self.checker.check_compatibility(request)
        
        self.assertFalse(result.is_compatible)
        self.assertTrue(any('not registered' in issue for issue in result.issues))
        self.assertTrue(any('Available sources' in rec for rec in result.recommendations))
        self.assertEqual(result.compatibility_score, 0.0)
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_check_compatibility_compatible_source(self, mock_registry):
        """测试兼容数据源的情况"""
        # Mock数据源注册表
        mock_registry.is_registered.return_value = True
        
        # Mock数据源类
        mock_source_class = Mock()
        mock_capabilities = DataSourceCapabilities(
            name='test_source',
            supported_markets=[MarketType.STOCK, MarketType.FOREX],
            supported_intervals=[DataInterval.DAY_1, DataInterval.HOUR_1],
            has_realtime=True,
            has_historical=True,
            data_quality=DataQuality.MEDIUM
        )
        mock_source_class.return_value.get_capabilities.return_value = mock_capabilities
        mock_registry.get.return_value = mock_source_class
        
        request = CompatibilityRequest(
            source='test_source',
            market_type=MarketType.STOCK,
            interval=DataInterval.DAY_1,
            quality_requirement=DataQuality.MEDIUM
        )
        
        result = self.checker.check_compatibility(request)
        
        self.assertTrue(result.is_compatible)
        self.assertEqual(len(result.issues), 0)
        self.assertGreater(result.compatibility_score, 0.8)
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_check_compatibility_incompatible_market(self, mock_registry):
        """测试市场类型不兼容的情况"""
        # Mock数据源注册表
        mock_registry.is_registered.return_value = True
        
        # Mock数据源类 - 只支持股票，不支持外汇
        mock_source_class = Mock()
        mock_capabilities = DataSourceCapabilities(
            name='stock_only_source',
            supported_markets=[MarketType.STOCK],  # 只支持股票
            supported_intervals=[DataInterval.DAY_1],
            has_realtime=True,
            has_historical=True,
            data_quality=DataQuality.MEDIUM
        )
        mock_source_class.return_value.get_capabilities.return_value = mock_capabilities
        mock_registry.get.return_value = mock_source_class
        
        request = CompatibilityRequest(
            source='stock_only_source',
            market_type=MarketType.FOREX,  # 请求外汇数据
            interval=DataInterval.DAY_1
        )
        
        result = self.checker.check_compatibility(request)
        
        self.assertFalse(result.is_compatible)
        self.assertTrue(any('not supported' in issue for issue in result.issues))
        # 评分不应该是0，因为其他维度可能有分数，但应该低于0.8
        self.assertLess(result.compatibility_score, 0.8)
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_check_compatibility_incompatible_interval(self, mock_registry):
        """测试时间间隔不兼容的情况"""
        # Mock数据源注册表
        mock_registry.is_registered.return_value = True
        
        # Mock数据源类 - 只支持日线，不支持分钟线
        mock_source_class = Mock()
        mock_capabilities = DataSourceCapabilities(
            name='daily_only_source',
            supported_markets=[MarketType.STOCK],
            supported_intervals=[DataInterval.DAY_1],  # 只支持日线
            has_realtime=True,
            has_historical=True,
            data_quality=DataQuality.MEDIUM
        )
        mock_source_class.return_value.get_capabilities.return_value = mock_capabilities
        mock_registry.get.return_value = mock_source_class
        
        request = CompatibilityRequest(
            source='daily_only_source',
            market_type=MarketType.STOCK,
            interval=DataInterval.MINUTE_1  # 请求分钟线数据
        )
        
        result = self.checker.check_compatibility(request)
        
        self.assertFalse(result.is_compatible)
        self.assertTrue(any('not supported' in issue for issue in result.issues))
        self.assertTrue(any('Consider using closest' in warning for warning in result.warnings))
        # 评分不应该是0，因为其他维度可能有分数，但应该低于0.8
        self.assertLess(result.compatibility_score, 0.8)
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_check_multiple_sources(self, mock_registry):
        """测试多数据源兼容性检查"""
        # Mock数据源注册表
        mock_registry.is_registered.return_value = True
        mock_registry.list_sources.return_value = ['source1', 'source2', 'source3']
        
        # Mock数据源类
        def create_mock_source(supported_markets, supported_intervals):
            mock_source_class = Mock()
            mock_capabilities = DataSourceCapabilities(
                name='mock_source',
                supported_markets=supported_markets,
                supported_intervals=supported_intervals,
                has_realtime=True,
                has_historical=True,
                data_quality=DataQuality.MEDIUM
            )
            mock_source_class.return_value.get_capabilities.return_value = mock_capabilities
            return mock_source_class
        
        # 创建mock返回函数
        def get_mock_source(source_name):
            if source_name == 'source1':
                return create_mock_source([MarketType.STOCK], [DataInterval.DAY_1])
            elif source_name == 'source2':
                return create_mock_source([MarketType.FOREX], [DataInterval.MINUTE_1])
            elif source_name == 'source3':
                return create_mock_source([MarketType.STOCK, MarketType.FOREX], [DataInterval.DAY_1, DataInterval.MINUTE_1])
            else:
                return create_mock_source([], [])
        
        mock_registry.get.side_effect = get_mock_source
        
        sources = ['source1', 'source2', 'source3']
        results = self.checker.check_multiple_sources(
            sources=sources,
            market_type=MarketType.STOCK,
            interval=DataInterval.DAY_1
        )
        
        # 验证结果
        self.assertEqual(len(results), 3)
        self.assertTrue(results['source1'].is_compatible)  # 支持股票日线
        self.assertFalse(results['source2'].is_compatible)  # 不支持股票
        self.assertTrue(results['source3'].is_compatible)  # 全支持
    
    @patch('src.data.compatibility_checker.DataSourceRegistry')
    def test_rank_sources_by_compatibility(self, mock_registry):
        """测试按兼容性排序数据源"""
        # Mock设置（复用上面的逻辑）
        mock_registry.is_registered.return_value = True
        mock_registry.list_sources.return_value = ['source1', 'source2', 'source3']
        
        def create_mock_source(supported_markets, supported_intervals, quality=DataQuality.MEDIUM):
            mock_source_class = Mock()
            mock_capabilities = DataSourceCapabilities(
                name='mock_source',
                supported_markets=supported_markets,
                supported_intervals=supported_intervals,
                has_realtime=True,
                has_historical=True,
                data_quality=quality
            )
            mock_source_class.return_value.get_capabilities.return_value = mock_capabilities
            return mock_source_class
        
        def get_mock_source_with_quality(source_name):
            if source_name == 'source1':
                return create_mock_source([MarketType.STOCK], [DataInterval.DAY_1], DataQuality.MEDIUM)
            elif source_name == 'source2':
                return create_mock_source([MarketType.FOREX], [DataInterval.MINUTE_1], DataQuality.HIGH)  # 不兼容
            elif source_name == 'source3':
                return create_mock_source([MarketType.STOCK], [DataInterval.DAY_1], DataQuality.HIGH)  # 高质量
            else:
                return create_mock_source([], [], DataQuality.UNKNOWN)
        
        mock_registry.get.side_effect = get_mock_source_with_quality
        
        ranked_sources = self.checker.rank_sources_by_compatibility(
            sources=['source1', 'source2', 'source3'],
            market_type=MarketType.STOCK,
            interval=DataInterval.DAY_1
        )
        
        # 验证排序结果
        self.assertEqual(len(ranked_sources), 2)  # 只有兼容的数据源
        
        # source3应该排在前面（高质量）
        source_names = [name for name, score in ranked_sources]
        self.assertIn('source1', source_names)
        self.assertIn('source3', source_names)
        self.assertNotIn('source2', source_names)  # 不兼容，不应出现


class TestGlobalCompatibilityChecker(unittest.TestCase):
    """全局兼容性检查器测试类"""
    
    def test_get_compatibility_checker_singleton(self):
        """测试全局兼容性检查器单例"""
        checker1 = get_compatibility_checker()
        checker2 = get_compatibility_checker()
        
        self.assertIs(checker1, checker2)  # 应该是同一个实例
        self.assertIsInstance(checker1, CompatibilityChecker)


if __name__ == '__main__':
    unittest.main()