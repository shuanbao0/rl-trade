"""
Test RewardContext and related classes
"""

import unittest
import numpy as np
from datetime import datetime

from src.rewards.core.reward_context import (
    RewardContext, RewardResult, RewardContextBuilder,
    create_simple_context, create_forex_context
)


class TestRewardContext(unittest.TestCase):
    """测试RewardContext类"""
    
    def test_basic_creation(self):
        """测试基础创建"""
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100
        )
        
        self.assertEqual(context.portfolio_value, 10500.0)
        self.assertEqual(context.action, 0.5)
        self.assertEqual(context.current_price, 1.2345)
        self.assertEqual(context.step, 100)
        self.assertIsNotNone(context.timestamp)
    
    def test_return_pct_calculation(self):
        """测试收益率计算"""
        context = RewardContext(
            portfolio_value=11000.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_info={'initial_balance': 10000.0}
        )
        
        return_pct = context.get_return_pct()
        self.assertEqual(return_pct, 10.0)  # (11000 - 10000) / 10000 * 100
    
    def test_step_return_calculation(self):
        """测试步骤收益率计算"""
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([9000, 9500, 10000, 10500])
        )
        
        step_return = context.get_step_return()
        self.assertAlmostEqual(step_return, 0.05, places=4)  # (10500 - 10000) / 10000
    
    def test_price_change_calculation(self):
        """测试价格变化计算"""
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            price_history=np.array([1.2000, 1.2100, 1.2200, 1.2345])
        )
        
        price_change = context.get_price_change()
        self.assertAlmostEqual(price_change, 0.0145, places=4)  # 1.2345 - 1.2200
    
    def test_volatility_calculation(self):
        """测试波动率计算"""
        prices = np.array([1.20, 1.21, 1.22, 1.21, 1.23, 1.22, 1.24, 1.25, 1.24, 1.26])
        
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.26,
            step=100,
            price_history=prices
        )
        
        volatility = context.get_volatility(window=10)
        self.assertGreater(volatility, 0)
        self.assertIsInstance(volatility, float)
    
    def test_sufficient_history_check(self):
        """测试历史数据充足性检查"""
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([9000, 9500, 10000, 10500])
        )
        
        self.assertTrue(context.has_sufficient_history(3))
        self.assertFalse(context.has_sufficient_history(10))
    
    def test_to_dict_conversion(self):
        """测试字典转换"""
        context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            market_type='forex',
            granularity='5min'
        )
        
        context_dict = context.to_dict()
        
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict['portfolio_value'], 10500.0)
        self.assertEqual(context_dict['action'], 0.5)
        self.assertEqual(context_dict['market_type'], 'forex')
        self.assertEqual(context_dict['granularity'], '5min')


class TestRewardResult(unittest.TestCase):
    """测试RewardResult类"""
    
    def test_basic_creation(self):
        """测试基础创建"""
        result = RewardResult(
            reward_value=1.5,
            components={'base': 1.0, 'bonus': 0.5},
            computation_time=0.001
        )
        
        self.assertEqual(result.reward_value, 1.5)
        self.assertEqual(result.components['base'], 1.0)
        self.assertEqual(result.computation_time, 0.001)
    
    def test_to_dict_conversion(self):
        """测试字典转换"""
        result = RewardResult(
            reward_value=1.5,
            components={'base': 1.0, 'bonus': 0.5},
            computation_time=0.001
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['reward_value'], 1.5)
        self.assertEqual(result_dict['computation_time'], 0.001)


class TestRewardContextBuilder(unittest.TestCase):
    """测试RewardContextBuilder类"""
    
    def test_builder_pattern(self):
        """测试建造者模式"""
        context = (RewardContextBuilder()
                  .with_portfolio_value(10500.0)
                  .with_action(0.5)
                  .with_price(1.2345)
                  .with_step(100)
                  .with_market_info('forex', '5min')
                  .build())
        
        self.assertEqual(context.portfolio_value, 10500.0)
        self.assertEqual(context.action, 0.5)
        self.assertEqual(context.current_price, 1.2345)
        self.assertEqual(context.step, 100)
        self.assertEqual(context.market_type, 'forex')
        self.assertEqual(context.granularity, '5min')
    
    def test_builder_with_history(self):
        """测试建造者模式与历史数据"""
        price_history = np.array([1.20, 1.21, 1.22, 1.2345])
        portfolio_history = np.array([10000, 10100, 10300, 10500])
        
        context = (RewardContextBuilder()
                  .with_portfolio_value(10500.0)
                  .with_action(0.5)
                  .with_price(1.2345)
                  .with_step(100)
                  .with_history(price_history, portfolio_history)
                  .build())
        
        self.assertIsNotNone(context.price_history)
        self.assertIsNotNone(context.portfolio_history)
        self.assertEqual(len(context.price_history), 4)
        self.assertEqual(len(context.portfolio_history), 4)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def test_create_simple_context(self):
        """测试简单上下文创建"""
        context = create_simple_context(10500.0, 0.5, 1.2345, 100)
        
        self.assertEqual(context.portfolio_value, 10500.0)
        self.assertEqual(context.action, 0.5)
        self.assertEqual(context.current_price, 1.2345)
        self.assertEqual(context.step, 100)
    
    def test_create_forex_context(self):
        """测试外汇上下文创建"""
        context = create_forex_context(
            10500.0, 0.5, 1.2345, 100,
            pip_size=0.0001, spread=2, leverage=100,
            granularity='1min'
        )
        
        self.assertEqual(context.portfolio_value, 10500.0)
        self.assertEqual(context.market_type, 'forex')
        self.assertEqual(context.granularity, '1min')
        self.assertEqual(context.metadata['pip_size'], 0.0001)
        self.assertEqual(context.metadata['spread'], 2)
        self.assertEqual(context.metadata['leverage'], 100)


if __name__ == '__main__':
    unittest.main()