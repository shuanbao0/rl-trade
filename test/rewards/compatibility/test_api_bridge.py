"""
Test APIBridge
"""

import unittest
from unittest.mock import MagicMock

from src.rewards.compatibility.api_bridge import APIBridge, LegacyFunctionWrapper
from src.rewards.compatibility.legacy_adapter import LegacyRewardAdapter
from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward


class MockReward(BaseReward):
    """测试用的奖励函数"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.name = 'mock_reward'
    
    def calculate(self, context: RewardContext) -> float:
        return context.portfolio_value * 0.001
    
    def get_info(self):
        return {
            'name': 'mock_reward',
            'type': 'test',
            'description': 'Mock reward for testing'
        }


class MockLegacyFunction:
    """模拟旧版本奖励函数"""
    
    def __init__(self):
        self.name = 'mock_legacy'
    
    def calculate_reward(self, portfolio_value, action, price, portfolio_info, trade_info, step):
        return portfolio_value * 0.002
    
    def get_reward_info(self):
        return {
            'name': 'mock_legacy',
            'type': 'legacy',
            'description': 'Mock legacy reward'
        }


class TestAPIBridge(unittest.TestCase):
    """测试API桥接器"""
    
    def setUp(self):
        """设置测试环境"""
        self.bridge = APIBridge()
        
        # 注册测试用的奖励函数
        self.bridge.register_reward('mock_reward', MockReward, ['mock', 'test'])
    
    def test_bridge_initialization(self):
        """测试桥接器初始化"""
        self.assertIsNotNone(self.bridge.converter)
        self.assertIsInstance(self.bridge._reward_registry, dict)
        self.assertIsInstance(self.bridge._legacy_aliases, dict)
        
        # 检查是否注册了默认映射
        self.assertIn('simple', self.bridge._legacy_aliases)
        self.assertIn('profit', self.bridge._legacy_aliases)
    
    def test_register_reward(self):
        """测试注册奖励函数"""
        self.bridge.register_reward('custom_reward', MockReward, ['custom', 'test_custom'])
        
        # 检查是否正确注册
        self.assertIn('custom_reward', self.bridge._reward_registry)
        self.assertEqual(self.bridge._reward_registry['custom_reward'], MockReward)
        
        # 检查别名
        self.assertIn('custom', self.bridge._legacy_aliases)
        self.assertIn('test_custom', self.bridge._legacy_aliases)
        self.assertEqual(self.bridge._legacy_aliases['custom'], 'custom_reward')
    
    def test_create_reward_new_mode(self):
        """测试创建新版本奖励函数"""
        reward = self.bridge.create_reward('mock_reward', legacy_mode=False)
        
        self.assertIsInstance(reward, MockReward)
        self.assertIsInstance(reward, BaseReward)
    
    def test_create_reward_legacy_mode(self):
        """测试创建旧版本兼容奖励函数"""
        reward = self.bridge.create_reward('mock_reward', legacy_mode=True)
        
        self.assertIsInstance(reward, LegacyRewardAdapter)
        self.assertIsInstance(reward.reward_function, MockReward)
    
    def test_create_legacy_reward(self):
        """测试创建旧版本奖励函数的快捷方法"""
        reward = self.bridge.create_legacy_reward('mock_reward')
        
        self.assertIsInstance(reward, LegacyRewardAdapter)
        self.assertIsInstance(reward.reward_function, MockReward)
    
    def test_create_reward_with_alias(self):
        """测试通过别名创建奖励函数"""
        reward = self.bridge.create_reward('mock')  # 使用别名
        
        self.assertIsInstance(reward, MockReward)
    
    def test_create_reward_unknown_type(self):
        """测试创建未知类型的奖励函数"""
        with self.assertRaises(ValueError):
            self.bridge.create_reward('unknown_reward_type')
    
    def test_get_available_rewards(self):
        """测试获取可用奖励函数列表"""
        available = self.bridge.get_available_rewards()
        
        self.assertIsInstance(available, list)
        self.assertIn('mock_reward', available)
        
        # 应该包含注册的奖励函数
        self.assertGreater(len(available), 0)
    
    def test_get_reward_info(self):
        """测试获取奖励函数信息"""
        info = self.bridge.get_reward_info('mock_reward')
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['name'], 'mock_reward')
        self.assertEqual(info['type'], 'test')
        self.assertIn('description', info)
    
    def test_get_reward_info_unknown_type(self):
        """测试获取未知奖励函数信息"""
        with self.assertRaises(ValueError):
            self.bridge.get_reward_info('unknown_type')
    
    def test_wrap_legacy_function(self):
        """测试包装旧版本函数"""
        legacy_func = MockLegacyFunction()
        wrapped = self.bridge.wrap_legacy_function(legacy_func)
        
        self.assertIsInstance(wrapped, LegacyFunctionWrapper)
        self.assertEqual(wrapped.legacy_function, legacy_func)
    
    def test_create_context_from_env(self):
        """测试从环境创建上下文"""
        # 创建模拟环境
        mock_env = type('MockEnv', (), {
            'portfolio_value': 10500.0,
            'action': 0.5,
            'price': 1.2,
            'step_count': 10
        })()
        
        context = self.bridge.create_context_from_env(mock_env)
        
        self.assertIsInstance(context, RewardContext)
        self.assertEqual(context.portfolio_value, 10500.0)
        self.assertEqual(context.action, 0.5)
        self.assertEqual(context.current_price, 1.2)
        self.assertEqual(context.step, 10)
    
    def test_create_context_with_custom_values(self):
        """测试使用自定义值创建上下文"""
        mock_env = type('MockEnv', (), {})()
        
        context = self.bridge.create_context_from_env(
            mock_env,
            portfolio_value=12000.0,
            action=0.8,
            price=1.5
        )
        
        self.assertEqual(context.portfolio_value, 12000.0)
        self.assertEqual(context.action, 0.8)
        self.assertEqual(context.current_price, 1.5)
    
    def test_resolve_reward_type(self):
        """测试解析奖励函数类型"""
        # 测试注册的类型
        resolved = self.bridge._resolve_reward_type('mock_reward')
        self.assertEqual(resolved, MockReward)
        
        # 测试别名
        resolved_alias = self.bridge._resolve_reward_type('mock')
        self.assertEqual(resolved_alias, MockReward)
        
        # 测试未知类型
        resolved_unknown = self.bridge._resolve_reward_type('unknown')
        self.assertIsNone(resolved_unknown)
    
    def test_to_camel_case(self):
        """测试下划线转驼峰命名"""
        self.assertEqual(self.bridge._to_camel_case('simple_return'), 'SimpleReturn')
        self.assertEqual(self.bridge._to_camel_case('profit_loss'), 'ProfitLoss')
        self.assertEqual(self.bridge._to_camel_case('risk_adjusted'), 'RiskAdjusted')


class TestLegacyFunctionWrapper(unittest.TestCase):
    """测试旧版本函数包装器"""
    
    def setUp(self):
        """设置测试环境"""
        from src.rewards.compatibility.context_converter import ContextConverter
        self.converter = ContextConverter()
        self.legacy_function = MockLegacyFunction()
        self.wrapper = LegacyFunctionWrapper(self.legacy_function, self.converter)
    
    def test_wrapper_initialization(self):
        """测试包装器初始化"""
        self.assertEqual(self.wrapper.legacy_function, self.legacy_function)
        self.assertEqual(self.wrapper.converter, self.converter)
        self.assertEqual(self.wrapper.name, 'mock_legacy')
    
    def test_calculate_method(self):
        """测试计算方法"""
        context = RewardContext(
            portfolio_value=10000.0,
            action=0.5,
            current_price=1.0,
            step=0
        )
        
        result = self.wrapper.calculate(context)
        
        # MockLegacyFunction.calculate_reward返回portfolio_value * 0.002
        expected = 10000.0 * 0.002
        self.assertEqual(result, expected)
    
    def test_get_info_method(self):
        """测试获取信息方法"""
        info = self.wrapper.get_info()
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['name'], 'mock_legacy')
        self.assertEqual(info['type'], 'legacy')
    
    def test_wrapper_with_callable_function(self):
        """测试包装普通可调用函数"""
        def simple_func(portfolio_value, action, price):
            return portfolio_value * 0.001
        
        wrapper = LegacyFunctionWrapper(simple_func, self.converter)
        
        context = RewardContext(
            portfolio_value=5000.0,
            action=0.3,
            current_price=1.1,
            step=5
        )
        
        result = wrapper.calculate(context)
        expected = 5000.0 * 0.001
        self.assertEqual(result, expected)
    
    def test_wrapper_with_reward_method(self):
        """测试包装有reward方法的函数"""
        class OldRewardWithRewardMethod:
            def reward(self, env):
                return env.portfolio.net_worth * 0.001
        
        old_reward = OldRewardWithRewardMethod()
        wrapper = LegacyFunctionWrapper(old_reward, self.converter)
        
        context = RewardContext(
            portfolio_value=8000.0,
            action=0.2,
            current_price=1.3,
            step=3
        )
        
        result = wrapper.calculate(context)
        expected = 8000.0 * 0.001
        self.assertEqual(result, expected)
    
    def test_wrapper_error_handling(self):
        """测试包装器错误处理"""
        class ErrorFunction:
            def calculate_reward(self, *args):
                raise Exception("Test error")
        
        error_func = ErrorFunction()
        wrapper = LegacyFunctionWrapper(error_func, self.converter)
        
        context = RewardContext(
            portfolio_value=1000.0,
            action=0.1,
            current_price=1.0,
            step=1
        )
        
        # 应该返回0而不是抛出异常
        result = wrapper.calculate(context)
        self.assertEqual(result, 0.0)


if __name__ == '__main__':
    unittest.main()