"""
Test LegacyRewardAdapter
"""

import unittest
import numpy as np

from src.rewards.functions import SimpleReturnReward
from src.rewards.compatibility.legacy_adapter import LegacyRewardAdapter


class MockTensorTradeEnv:
    """模拟TensorTrade环境"""
    
    def __init__(self, portfolio_value=10000.0, action=0.5, price=1.0):
        self.portfolio = MockPortfolio(portfolio_value)
        self.action = action
        self.price = price
        self.current_price = price
        self.step_count = 0


class MockPortfolio:
    """模拟Portfolio对象"""
    
    def __init__(self, net_worth=10000.0):
        self.net_worth = net_worth
        self.initial_balance = 10000.0
        self.balance = net_worth * 0.5
        self.total_value = net_worth


class TestLegacyRewardAdapter(unittest.TestCase):
    """测试旧版奖励函数适配器"""
    
    def setUp(self):
        """设置测试环境"""
        self.reward_function = SimpleReturnReward()
        self.adapter = LegacyRewardAdapter(self.reward_function)
    
    def test_adapter_initialization(self):
        """测试适配器初始化"""
        self.assertIsNotNone(self.adapter.reward_function)
        self.assertIsNotNone(self.adapter.converter)
        self.assertEqual(self.adapter.step_count, 0)
        self.assertEqual(self.adapter.episode_count, 0)
        self.assertEqual(self.adapter.portfolio_history, [])
        self.assertEqual(self.adapter.reward_history, [])
    
    def test_reward_method_with_env(self):
        """测试通过环境计算奖励"""
        env = MockTensorTradeEnv(portfolio_value=10500.0, action=0.5, price=1.2)
        
        reward_value = self.adapter.reward(env)
        
        self.assertIsInstance(reward_value, float)
        self.assertEqual(self.adapter.step_count, 1)
        self.assertEqual(len(self.adapter.portfolio_history), 1)
        self.assertEqual(len(self.adapter.reward_history), 1)
        self.assertEqual(self.adapter.portfolio_history[0], 10500.0)
    
    def test_get_reward_method_with_portfolio(self):
        """测试通过Portfolio对象计算奖励"""
        portfolio = MockPortfolio(net_worth=11000.0)
        
        reward_value = self.adapter.get_reward(portfolio)
        
        self.assertIsInstance(reward_value, float)
        self.assertEqual(self.adapter.step_count, 1)
        self.assertEqual(len(self.adapter.portfolio_history), 1)
        self.assertEqual(self.adapter.portfolio_history[0], 11000.0)
    
    def test_calculate_reward_method(self):
        """测试旧版calculate_reward方法"""
        portfolio_info = {'initial_balance': 10000.0}
        trade_info = {}
        
        reward_value = self.adapter.calculate_reward(
            portfolio_value=10200.0,
            action=0.3,
            price=1.1,
            portfolio_info=portfolio_info,
            trade_info=trade_info,
            step=5
        )
        
        self.assertIsInstance(reward_value, float)
    
    def test_get_reward_info_method(self):
        """测试获取奖励信息方法"""
        info = self.adapter.get_reward_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('description', info)
        self.assertIn('category', info)
        self.assertIn('type', info)
        self.assertIn('step_count', info)
        self.assertIn('episode_count', info)
        self.assertIn('full_info', info)
        
        # 检查是否包含新版本的完整信息
        self.assertIsInstance(info['full_info'], dict)
    
    def test_reset_method(self):
        """测试重置方法"""
        # 先进行一些操作
        env = MockTensorTradeEnv()
        self.adapter.reward(env)
        self.adapter.reward(env)
        
        self.assertEqual(self.adapter.step_count, 2)
        self.assertEqual(len(self.adapter.portfolio_history), 2)
        
        # 重置
        self.adapter.reset()
        
        self.assertEqual(self.adapter.step_count, 0)
        self.assertEqual(self.adapter.episode_count, 1)
        self.assertEqual(len(self.adapter.portfolio_history), 0)
        self.assertEqual(len(self.adapter.reward_history), 0)
    
    def test_direct_call(self):
        """测试直接调用适配器"""
        env = MockTensorTradeEnv(portfolio_value=10300.0)
        
        # 测试单参数调用
        reward1 = self.adapter(env)
        self.assertIsInstance(reward1, float)
        
        # 测试多参数调用
        reward2 = self.adapter(
            portfolio_value=10400.0,
            action=0.2,
            price=1.05,
            portfolio_info={},
            trade_info={},
            step=10
        )
        self.assertIsInstance(reward2, float)
    
    def test_extract_portfolio_value(self):
        """测试提取投资组合价值"""
        # 测试标准环境
        env1 = MockTensorTradeEnv(portfolio_value=15000.0)
        value1 = self.adapter._extract_portfolio_value(env1)
        self.assertEqual(value1, 15000.0)
        
        # 测试没有portfolio属性的环境
        env2 = type('SimpleEnv', (), {'net_worth': 12000.0})()
        value2 = self.adapter._extract_portfolio_value(env2)
        self.assertEqual(value2, 12000.0)
        
        # 测试没有任何相关属性的环境
        env3 = type('EmptyEnv', (), {})()
        value3 = self.adapter._extract_portfolio_value(env3)
        self.assertEqual(value3, self.adapter.initial_balance)
    
    def test_extract_action(self):
        """测试提取动作"""
        # 测试单值动作
        env1 = type('Env1', (), {'action': 0.7})()
        action1 = self.adapter._extract_action(env1)
        self.assertEqual(action1, 0.7)
        
        # 测试数组动作
        env2 = type('Env2', (), {'action': [0.3, 0.8]})()
        action2 = self.adapter._extract_action(env2)
        self.assertEqual(action2, 0.3)
        
        # 测试numpy数组动作
        env3 = type('Env3', (), {'action': np.array([0.6])})()
        action3 = self.adapter._extract_action(env3)
        self.assertEqual(action3, 0.6)
        
        # 测试没有动作的环境
        env4 = type('Env4', (), {})()
        action4 = self.adapter._extract_action(env4)
        self.assertEqual(action4, 0.0)
    
    def test_extract_price(self):
        """测试提取价格"""
        # 测试标准价格
        env1 = type('Env1', (), {'price': 1.5})()
        price1 = self.adapter._extract_price(env1)
        self.assertEqual(price1, 1.5)
        
        # 测试current_price
        env2 = type('Env2', (), {'current_price': 2.0})()
        price2 = self.adapter._extract_price(env2)
        self.assertEqual(price2, 2.0)
        
        # 测试没有价格的环境
        env3 = type('Env3', (), {})()
        price3 = self.adapter._extract_price(env3)
        self.assertEqual(price3, 1.0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 创建会抛出异常的环境
        class ErrorEnv:
            @property
            def portfolio(self):
                raise Exception("Test error")
        
        env = ErrorEnv()
        
        # 应该不会抛出异常，而是返回0奖励
        reward_value = self.adapter.reward(env)
        self.assertEqual(reward_value, 0.0)
    
    def test_property_compatibility(self):
        """测试属性兼容性"""
        # 测试previous_value属性
        self.assertEqual(self.adapter.previous_value, self.adapter.initial_balance)
        
        # 添加一些历史数据
        self.adapter.update_history(10500.0)
        self.assertEqual(self.adapter.previous_value, 10500.0)
    
    def test_update_history_method(self):
        """测试更新历史方法"""
        initial_step = self.adapter.step_count
        
        self.adapter.update_history(10200.0)
        
        self.assertEqual(len(self.adapter.portfolio_history), 1)
        self.assertEqual(self.adapter.portfolio_history[0], 10200.0)
        self.assertEqual(self.adapter.step_count, initial_step + 1)


if __name__ == '__main__':
    unittest.main()