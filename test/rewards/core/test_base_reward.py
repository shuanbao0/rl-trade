"""
Test BaseReward and related classes
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import time

from src.rewards.core.reward_context import RewardContext
from src.rewards.core.base_reward import BaseReward, RewardPlugin, SimpleRewardMixin, HistoryAwareRewardMixin


class MockReward(BaseReward):
    """测试用的简单奖励函数"""
    
    def __init__(self, **config):
        super().__init__(**config)
        self.calculation_count = 0
    
    def calculate(self, context: RewardContext) -> float:
        self.calculation_count += 1
        return context.get_step_return() * 10.0
    
    def get_info(self) -> dict:
        return {
            'name': self.name,
            'type': 'mock',
            'description': 'Mock reward for testing',
            'calculation_count': self.calculation_count
        }


class MockPlugin(RewardPlugin):
    """测试用的插件"""
    
    def __init__(self):
        self.before_calls = 0
        self.after_calls = 0
        self.error_calls = 0
    
    def before_calculate(self, context: RewardContext) -> RewardContext:
        self.before_calls += 1
        return context
    
    def after_calculate(self, context: RewardContext, reward: float) -> float:
        self.after_calls += 1
        return reward * 1.1  # 给奖励加10%
    
    def on_error(self, context: RewardContext, error: Exception):
        self.error_calls += 1


class ErrorReward(BaseReward):
    """会抛出错误的奖励函数"""
    
    def calculate(self, context: RewardContext) -> float:
        raise ValueError("Test error")
    
    def get_info(self) -> dict:
        return {'name': 'ErrorReward', 'type': 'error'}


class TestBaseReward(unittest.TestCase):
    """测试BaseReward基类"""
    
    def setUp(self):
        """测试前设置"""
        self.context = RewardContext(
            portfolio_value=10500.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([10000, 10500])
        )
    
    def test_basic_reward_creation(self):
        """测试基础奖励函数创建"""
        reward = MockReward(name="TestReward")
        
        self.assertEqual(reward.name, "TestReward")
        self.assertEqual(reward.version, "2.0.0")
        self.assertTrue(reward.is_initialized)
        self.assertEqual(reward.call_count, 0)
    
    def test_reward_calculation(self):
        """测试奖励计算"""
        reward = MockReward()
        
        result = reward(self.context)
        
        self.assertIsInstance(result, float)
        self.assertEqual(reward.call_count, 1)
        self.assertGreaterEqual(reward.total_compute_time, 0)  # 允许为0（计算太快）
        self.assertEqual(reward.last_reward, result)
    
    def test_plugin_system(self):
        """测试插件系统"""
        reward = MockReward()
        plugin = MockPlugin()
        
        reward.add_plugin(plugin)
        
        result = reward(self.context)
        
        self.assertEqual(plugin.before_calls, 1)
        self.assertEqual(plugin.after_calls, 1)
        # 结果应该被插件修改（×1.1）
        expected = self.context.get_step_return() * 10.0 * 1.1
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_error_handling_with_default_reward(self):
        """测试错误处理（返回默认奖励）"""
        reward = ErrorReward(default_reward=0.5, raise_on_error=False)
        
        result = reward(self.context)
        
        self.assertEqual(result, 0.5)  # 应该返回默认奖励
        self.assertEqual(reward.error_count, 1)
    
    def test_error_handling_with_exception(self):
        """测试错误处理（抛出异常）"""
        reward = ErrorReward(raise_on_error=True)
        
        with self.assertRaises(ValueError):
            reward(self.context)
        
        self.assertEqual(reward.error_count, 1)
    
    def test_plugin_error_handling(self):
        """测试插件错误处理"""
        reward = MockReward()
        plugin = MockPlugin()
        
        reward.add_plugin(plugin)
        
        # 模拟插件错误
        with patch.object(plugin, 'before_calculate', side_effect=Exception("Plugin error")):
            result = reward(self.context)  # 应该仍然能计算奖励
        
        self.assertIsInstance(result, float)
        self.assertEqual(plugin.error_calls, 0)  # before_calculate错误不会调用on_error
    
    def test_reward_validation(self):
        """测试奖励值验证"""
        class InvalidReward(BaseReward):
            def calculate(self, context):
                return float('nan')  # 返回无效值
            
            def get_info(self):
                return {'name': 'InvalidReward'}
        
        reward = InvalidReward()
        result = reward(self.context)
        
        self.assertEqual(result, 0.0)  # 应该被修正为0.0
    
    def test_reward_range_clipping(self):
        """测试奖励值范围限制"""
        class HighReward(BaseReward):
            def calculate(self, context):
                return 100.0  # 返回很高的值
            
            def get_info(self):
                return {'name': 'HighReward'}
        
        reward = HighReward(reward_range=(-1.0, 1.0))
        result = reward(self.context)
        
        self.assertEqual(result, 1.0)  # 应该被限制在范围内
    
    def test_reset_functionality(self):
        """测试重置功能"""
        reward = MockReward()
        plugin = MockPlugin()
        reward.add_plugin(plugin)
        
        # 先计算几次
        reward(self.context)
        reward(self.context)
        
        self.assertEqual(reward.call_count, 2)
        self.assertEqual(plugin.before_calls, 2)
        
        # 重置
        reward.reset()
        
        self.assertEqual(reward.call_count, 0)
        self.assertEqual(reward.total_compute_time, 0.0)
        self.assertEqual(reward.error_count, 0)
        self.assertIsNone(reward.last_context)
        self.assertIsNone(reward.last_reward)
    
    def test_performance_stats(self):
        """测试性能统计"""
        reward = MockReward()
        
        # 计算几次
        for _ in range(3):
            reward(self.context)
        
        stats = reward.get_performance_stats()
        
        self.assertEqual(stats['call_count'], 3)
        self.assertGreaterEqual(stats['total_compute_time'], 0)
        self.assertGreaterEqual(stats['avg_compute_time'], 0)
        self.assertEqual(stats['error_count'], 0)
        self.assertEqual(stats['error_rate'], 0.0)
    
    def test_context_validation(self):
        """测试上下文验证"""
        reward = MockReward()
        
        # 有效上下文
        valid_context = RewardContext(
            portfolio_value=1000.0,
            action=0.5,
            current_price=1.0,
            step=1
        )
        self.assertTrue(reward.validate_context(valid_context))
        
        # 无效上下文（缺少必需字段）
        invalid_context = RewardContext(
            portfolio_value=None,
            action=0.5,
            current_price=1.0,
            step=1
        )
        self.assertFalse(reward.validate_context(invalid_context))
    
    def test_calculate_with_result(self):
        """测试带详细结果的计算"""
        reward = MockReward()
        
        result = reward.calculate_with_result(self.context)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result.reward_value, float)
        self.assertGreaterEqual(result.computation_time, 0)
        self.assertIn('total', result.components)
        self.assertIn('call_count', result.metadata)


class TestSimpleRewardMixin(unittest.TestCase):
    """测试SimpleRewardMixin"""
    
    def setUp(self):
        self.context = RewardContext(
            portfolio_value=11000.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([10000, 10500, 11000]),
            portfolio_info={'initial_balance': 10000.0}
        )
    
    def test_mixin_functionality(self):
        """测试混入功能"""
        
        class TestReward(BaseReward, SimpleRewardMixin):
            def calculate(self, context):
                return self.get_step_return(context) + self.get_total_return(context) * 0.1
            
            def get_info(self):
                return {'name': 'TestReward'}
        
        reward = TestReward()
        result = reward(self.context)
        
        # 应该包含步骤收益率和总收益率的组合
        expected_step = (11000 - 10500) / 10500  # 步骤收益率
        expected_total = (11000 - 10000) / 10000 * 100  # 总收益率百分比
        expected = expected_step + expected_total * 0.1
        
        self.assertAlmostEqual(result, expected, places=5)


class TestHistoryAwareRewardMixin(unittest.TestCase):
    """测试HistoryAwareRewardMixin"""
    
    def setUp(self):
        self.context = RewardContext(
            portfolio_value=11000.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([9000, 9500, 10000, 10500, 11000])
        )
    
    def test_history_aware_functionality(self):
        """测试历史感知功能"""
        
        class TestReward(BaseReward, HistoryAwareRewardMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def calculate(self, context):
                if not self.has_sufficient_history(context):
                    return 0.0
                
                returns = self.get_returns_series(context)
                return np.mean(returns) if len(returns) > 0 else 0.0
            
            def get_info(self):
                return {'name': 'TestReward'}
        
        reward = TestReward(min_history_steps=3)
        result = reward(self.context)
        
        self.assertIsInstance(result, float)
        # 应该能够计算历史收益率的平均值
        
    def test_insufficient_history(self):
        """测试历史数据不足的情况"""
        
        class TestReward(BaseReward, HistoryAwareRewardMixin):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def calculate(self, context):
                if not self.has_sufficient_history(context):
                    return 0.0
                return 1.0
            
            def get_info(self):
                return {'name': 'TestReward'}
        
        # 上下文历史数据不足
        insufficient_context = RewardContext(
            portfolio_value=11000.0,
            action=0.5,
            current_price=1.2345,
            step=100,
            portfolio_history=np.array([10500, 11000])  # 只有2个数据点
        )
        
        reward = TestReward(min_history_steps=5)  # 需要5个数据点
        result = reward(insufficient_context)
        
        self.assertEqual(result, 0.0)  # 应该返回0因为历史数据不足


if __name__ == '__main__':
    unittest.main()