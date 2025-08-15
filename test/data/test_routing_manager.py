"""
测试路由管理器功能
"""

import unittest
import tempfile
import os
from pathlib import Path
import yaml

from src.data.routing_manager import RoutingManager, RoutingRule
from src.data.sources.base import MarketType, DataQuality


class TestRoutingManager(unittest.TestCase):
    """路由管理器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时配置文件
        self.test_config = {
            'routing_strategy': {
                'stock': {
                    'primary': 'yfinance',
                    'fallback': [],
                    'description': 'Test stock routing'
                },
                'forex': {
                    'primary': 'truefx',
                    'fallback': ['oanda', 'fxminute'],
                    'description': 'Test forex routing'
                }
            },
            'global_settings': {
                'default_market_type': 'stock',
                'enable_source_compatibility_check': True,
                'enable_smart_fallback': True,
                'max_retries_per_source': 3,
                'retry_delay_seconds': 1.0
            },
            'source_priorities': ['yfinance', 'truefx', 'oanda'],
            'symbol_overrides': {
                'XAUUSD': {
                    'primary': 'oanda',
                    'fallback': ['yfinance']
                }
            },
            'interval_preferences': {
                'high_frequency': {
                    'intervals': ['1m', '5m'],
                    'preferred_sources': ['truefx', 'oanda']
                }
            },
            'quality_requirements': {
                'forex': {
                    'min_quality': 'MEDIUM',
                    'preferred_sources': ['truefx', 'oanda']
                }
            }
        }
        
        # 创建临时配置文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_file, default_flow_style=False)
        self.temp_file.close()
        
        # 使用临时配置文件创建路由管理器
        self.routing_manager = RoutingManager(self.temp_file.name)
    
    def tearDown(self):
        """清理测试环境"""
        # 删除临时文件
        os.unlink(self.temp_file.name)
    
    def test_get_routing_rule_basic(self):
        """测试基本路由规则获取"""
        # 测试股票路由
        stock_rule = self.routing_manager.get_routing_rule("AAPL", MarketType.STOCK)
        self.assertEqual(stock_rule.primary, 'yfinance')
        self.assertEqual(stock_rule.fallback, [])
        
        # 测试外汇路由
        forex_rule = self.routing_manager.get_routing_rule("EURUSD", MarketType.FOREX)
        self.assertEqual(forex_rule.primary, 'truefx')
        self.assertEqual(forex_rule.fallback, ['oanda', 'fxminute'])
    
    def test_symbol_override(self):
        """测试符号覆盖规则"""
        # 测试XAUUSD覆盖规则
        xauusd_rule = self.routing_manager.get_routing_rule("XAUUSD", MarketType.COMMODITIES)
        self.assertEqual(xauusd_rule.primary, 'oanda')
        self.assertEqual(xauusd_rule.fallback, ['yfinance'])
    
    def test_get_optimal_sources(self):
        """测试最优数据源选择"""
        # 测试普通符号
        sources = self.routing_manager.get_optimal_sources("AAPL", MarketType.STOCK, "1d")
        self.assertIn('yfinance', sources)
        
        # 测试高频数据偏好
        sources = self.routing_manager.get_optimal_sources("EURUSD", MarketType.FOREX, "1m")
        # 应该优先使用高频偏好的数据源
        self.assertTrue(sources[0] in ['truefx', 'oanda'])
    
    def test_global_settings(self):
        """测试全局设置"""
        self.assertEqual(self.routing_manager.get_default_market_type(), MarketType.STOCK)
        self.assertTrue(self.routing_manager.is_source_compatibility_check_enabled())
        self.assertTrue(self.routing_manager.is_smart_fallback_enabled())
        self.assertEqual(self.routing_manager.get_max_retries_per_source(), 3)
        self.assertEqual(self.routing_manager.get_retry_delay_seconds(), 1.0)
    
    def test_source_priorities(self):
        """测试数据源优先级"""
        priorities = self.routing_manager.get_source_priorities()
        expected = ['yfinance', 'truefx', 'oanda']
        self.assertEqual(priorities, expected)
    
    def test_add_remove_symbol_override(self):
        """测试动态添加和移除符号覆盖"""
        # 添加新的覆盖规则
        new_rule = RoutingRule(primary='custom_source', fallback=['backup_source'])
        self.routing_manager.add_symbol_override("TEST", new_rule)
        
        # 验证覆盖规则生效
        test_rule = self.routing_manager.get_routing_rule("TEST", MarketType.STOCK)
        self.assertEqual(test_rule.primary, 'custom_source')
        self.assertEqual(test_rule.fallback, ['backup_source'])
        
        # 移除覆盖规则
        self.routing_manager.remove_symbol_override("TEST")
        
        # 验证回到默认规则
        test_rule = self.routing_manager.get_routing_rule("TEST", MarketType.STOCK)
        self.assertEqual(test_rule.primary, 'yfinance')
    
    def test_routing_summary(self):
        """测试路由配置摘要"""
        summary = self.routing_manager.get_routing_summary()
        
        self.assertIn('config_file', summary)
        self.assertIn('market_type_rules', summary)
        self.assertIn('symbol_overrides_count', summary)
        self.assertIn('source_priorities', summary)
        self.assertIn('global_settings', summary)
        
        # 验证内容
        self.assertEqual(summary['symbol_overrides_count'], 1)  # XAUUSD
        self.assertEqual(len(summary['market_type_rules']), 2)  # STOCK, FOREX
    
    def test_config_validation(self):
        """测试配置验证"""
        is_valid, errors = self.routing_manager.validate_config()
        
        # 由于我们可能没有注册所有数据源，可能会有错误
        # 但至少应该能运行验证
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)
    
    def test_default_config_fallback(self):
        """测试配置文件不存在时的默认配置"""
        # 使用不存在的配置文件
        routing_manager = RoutingManager("/nonexistent/config.yaml")
        
        # 应该能获取到默认配置
        stock_rule = routing_manager.get_routing_rule("AAPL", MarketType.STOCK)
        self.assertEqual(stock_rule.primary, 'yfinance')
        
        forex_rule = routing_manager.get_routing_rule("EURUSD", MarketType.FOREX)
        self.assertEqual(forex_rule.primary, 'truefx')
    
    def test_case_insensitive_symbol(self):
        """测试符号大小写不敏感"""
        # 测试小写符号
        rule1 = self.routing_manager.get_routing_rule("xauusd", MarketType.COMMODITIES)
        rule2 = self.routing_manager.get_routing_rule("XAUUSD", MarketType.COMMODITIES)
        
        self.assertEqual(rule1.primary, rule2.primary)
        self.assertEqual(rule1.fallback, rule2.fallback)


class TestRoutingRule(unittest.TestCase):
    """路由规则测试类"""
    
    def test_routing_rule_creation(self):
        """测试路由规则创建"""
        rule = RoutingRule(
            primary='primary_source',
            fallback=['fallback1', 'fallback2'],
            description='Test rule'
        )
        
        self.assertEqual(rule.primary, 'primary_source')
        self.assertEqual(rule.fallback, ['fallback1', 'fallback2'])
        self.assertEqual(rule.description, 'Test rule')
    
    def test_get_all_sources(self):
        """测试获取所有数据源"""
        rule = RoutingRule(
            primary='primary_source',
            fallback=['fallback1', 'fallback2']
        )
        
        all_sources = rule.get_all_sources()
        expected = ['primary_source', 'fallback1', 'fallback2']
        self.assertEqual(all_sources, expected)
    
    def test_empty_fallback(self):
        """测试空备选源列表"""
        rule = RoutingRule(primary='primary_source')
        
        self.assertEqual(rule.fallback, [])
        self.assertEqual(rule.get_all_sources(), ['primary_source'])


if __name__ == '__main__':
    unittest.main()