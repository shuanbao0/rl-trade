"""
测试性能优化奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.performance_optimizer import PerformanceOptimizer


class TestPerformanceOptimizer:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.optimizer = PerformanceOptimizer(
            optimization_target='sharpe_ratio',
            learning_rate=0.01,
            momentum=0.9,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.optimizer.optimization_target == 'sharpe_ratio'
        assert self.optimizer.learning_rate == 0.01
        assert self.optimizer.momentum == 0.9
        assert self.optimizer.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.optimizer.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Performance Optimizer'
    
    def test_optimization_step(self):
        """测试优化步骤"""
        current_params = {'weight1': 0.5, 'weight2': 0.3}
        gradient = {'weight1': 0.1, 'weight2': -0.05}
        
        updated_params = self.optimizer._optimization_step(current_params, gradient)
        
        assert isinstance(updated_params, dict)
        assert 'weight1' in updated_params
        assert 'weight2' in updated_params
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.optimizer.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)