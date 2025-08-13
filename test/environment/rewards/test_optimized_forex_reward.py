"""
OptimizedForexReward 测试文件
测试集成到项目中的优化外汇奖励函数
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.environment.rewards.optimized_forex_reward import (
    OptimizedForexReward, 
    OptimizedForexRewardConfig,
    create_optimized_forex_reward
)


class TestOptimizedForexRewardConfig(unittest.TestCase):
    """测试OptimizedForexRewardConfig配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = OptimizedForexRewardConfig()
        
        self.assertEqual(config.return_weight, 1.0)
        self.assertEqual(config.risk_penalty, 0.1)
        self.assertEqual(config.transaction_cost, 0.0001)
        self.assertEqual(config.correlation_threshold, 0.8)
        self.assertEqual(config.stability_window, 20)
        self.assertEqual(config.clip_range, (-1.0, 1.0))
        self.assertTrue(config.volatility_adjustment)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = OptimizedForexRewardConfig(
            return_weight=2.0,
            risk_penalty=0.2,
            correlation_threshold=0.9
        )
        
        self.assertEqual(config.return_weight, 2.0)
        self.assertEqual(config.risk_penalty, 0.2)
        self.assertEqual(config.correlation_threshold, 0.9)


class TestOptimizedForexReward(unittest.TestCase):
    """测试OptimizedForexReward奖励函数"""
    
    def setUp(self):
        """测试前设置"""
        self.config = OptimizedForexRewardConfig()
        self.reward = OptimizedForexReward(
            initial_balance=10000,
            config=self.config,
            base_currency_pair="EURUSD"
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.reward.initial_balance, 10000)
        self.assertEqual(self.reward.currency_pair, "EURUSD")
        self.assertIsNotNone(self.reward.config)
        self.assertIsNone(self.reward.prev_portfolio_value)
        self.assertEqual(self.reward.step_count, 0)
        self.assertEqual(len(self.reward.returns_history), 0)
    
    def test_get_reward_info(self):
        """测试获取奖励函数信息"""
        info = self.reward.get_reward_info()
        
        self.assertEqual(info['name'], 'OptimizedForexReward')
        self.assertEqual(info['experiment'], '005')
        self.assertIn('key_improvements', info)
        self.assertIn('parameters', info)
        self.assertEqual(len(info['key_improvements']), 5)
    
    def test_first_reward_calculation(self):
        """测试首次奖励计算"""
        observation = {'close': 1.1000, 'ATR_14': 0.0012}
        info = {'portfolio_value': 10000.0}
        
        reward = self.reward.calculate_reward(
            observation=observation,
            action=0.5,
            info=info
        )
        
        self.assertEqual(reward, 0.0)  # 首次调用应返回0
        self.assertEqual(self.reward.prev_portfolio_value, 10000.0)
    
    def test_reward_calculation_with_profit(self):
        """测试有盈利时的奖励计算"""
        # 首次调用
        self.reward.calculate_reward(
            observation={'close': 1.1000},
            action=0.5,
            info={'portfolio_value': 10000.0}
        )
        
        # 第二次调用，有盈利
        reward = self.reward.calculate_reward(
            observation={'close': 1.1010, 'ATR_14': 0.0012},
            action=0.6,
            info={'portfolio_value': 10100.0}
        )
        
        self.assertGreater(reward, 0)  # 盈利时奖励应为正
        self.assertEqual(len(self.reward.returns_history), 1)
        self.assertGreater(self.reward.returns_history[0], 0)
    
    def test_reward_calculation_with_loss(self):
        """测试有亏损时的奖励计算"""
        # 首次调用
        self.reward.calculate_reward(
            observation={'close': 1.1000},
            action=0.5,
            info={'portfolio_value': 10000.0}
        )
        
        # 第二次调用，有亏损
        reward = self.reward.calculate_reward(
            observation={'close': 1.0990, 'ATR_14': 0.0015},
            action=0.4,
            info={'portfolio_value': 9900.0}
        )
        
        self.assertLess(reward, 0)  # 亏损时奖励应为负
        self.assertEqual(len(self.reward.returns_history), 1)
        self.assertLess(self.reward.returns_history[0], 0)
    
    def test_stability_controls(self):
        """测试数值稳定性控制"""
        # 测试正常值
        reward = self.reward._apply_stability_controls(0.5)
        self.assertEqual(reward, 0.5)
        
        # 测试异常大的值
        reward = self.reward._apply_stability_controls(100.0)
        self.assertLessEqual(abs(reward), 1.0)  # 经过修正后应该在合理范围内
        
        # 测试超出范围的值
        reward = self.reward._apply_stability_controls(-5.0)
        self.assertGreaterEqual(reward, self.config.clip_range[0])
        
        # 测试无穷大值
        reward = self.reward._apply_stability_controls(float('inf'))
        self.assertEqual(reward, self.config.clip_range[1])  # 应该被裁剪到上界
        
        # 测试NaN值
        reward = self.reward._apply_stability_controls(float('nan'))
        self.assertEqual(reward, 0.0)
    
    def test_reward_return_consistency_validation(self):
        """测试奖励-回报一致性验证"""
        # 添加足够的历史数据
        for i in range(25):
            self.reward.returns_history.append(0.01 * i)
            self.reward.rewards_history.append(0.01 * i)  # 高度相关
        
        # 更新相关性分数
        self.reward._update_correlation_score()
        
        # 验证一致性
        is_consistent = self.reward.validate_reward_return_consistency()
        self.assertTrue(is_consistent)
        
        # 测试数据不足的情况
        self.reward.returns_history = [0.1, 0.2]
        self.reward.rewards_history = [0.1, 0.2]
        is_consistent = self.reward.validate_reward_return_consistency()
        self.assertFalse(is_consistent)
    
    def test_diagnostics(self):
        """测试诊断功能"""
        # 测试数据不足时的诊断
        diagnostics = self.reward.get_diagnostics()
        self.assertEqual(diagnostics['status'], 'insufficient_data')
        
        # 添加足够数据后再测试
        self.reward.returns_history = [0.01, 0.02, 0.01]
        self.reward.rewards_history = [0.01, 0.02, 0.01]
        self.reward.portfolio_values = [10000, 10100, 10200]
        self.reward.step_count = 3
        
        diagnostics = self.reward.get_diagnostics()
        self.assertIn('correlation_score', diagnostics)
        self.assertIn('mean_return', diagnostics)
        self.assertIn('mean_reward', diagnostics)
        self.assertIn('consistency_check', diagnostics)


class TestOptimizedForexRewardFactory(unittest.TestCase):
    """测试工厂函数"""
    
    def test_create_with_default_config(self):
        """测试使用默认配置创建"""
        reward = create_optimized_forex_reward()
        self.assertIsInstance(reward, OptimizedForexReward)
        self.assertIsInstance(reward.config, OptimizedForexRewardConfig)
    
    def test_create_with_custom_config(self):
        """测试使用自定义配置创建"""
        config = {
            'return_weight': 2.0,
            'risk_penalty': 0.2,
            'initial_balance': 20000
        }
        
        reward = create_optimized_forex_reward(config)
        self.assertEqual(reward.config.return_weight, 2.0)
        self.assertEqual(reward.config.risk_penalty, 0.2)
        self.assertEqual(reward.initial_balance, 20000)


class TestOptimizedForexRewardIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_integration_with_reward_factory(self):
        """测试与奖励工厂的集成"""
        from src.environment.rewards import create_reward_function
        
        # 测试各种别名
        aliases = [
            'optimized_forex_reward',
            'experiment_005',
            'enhanced_forex',
            'reward_return_consistent'
        ]
        
        for alias in aliases:
            with self.subTest(alias=alias):
                reward = create_reward_function(alias, initial_balance=10000)
                self.assertIsInstance(reward, OptimizedForexReward)
    
    def test_train_model_integration(self):
        """测试与train_model.py的集成"""
        from src.environment.rewards import create_reward_function
        
        # 模拟train_model.py中的奖励函数创建
        reward_kwargs = {
            'initial_balance': 50000,
            'return_weight': 1.0,
            'risk_penalty': 0.1,
            'transaction_cost': 0.0001,
            'correlation_threshold': 0.8,
            'stability_window': 20,
            'volatility_adjustment': True,
            'clip_range': (-1.0, 1.0),
            'base_currency_pair': 'EURUSD'
        }
        
        reward = create_reward_function('optimized_forex_reward', **reward_kwargs)
        self.assertIsInstance(reward, OptimizedForexReward)
        self.assertEqual(reward.initial_balance, 50000)
        self.assertEqual(reward.currency_pair, 'EURUSD')


if __name__ == '__main__':
    unittest.main()