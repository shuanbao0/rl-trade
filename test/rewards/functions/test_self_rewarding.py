"""
测试自奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.self_rewarding import SelfRewardingReward


class TestSelfRewardingReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = SelfRewardingReward(
            exploration_weight=0.2,
            self_improvement_bonus=0.1,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.exploration_weight == 0.2
        assert self.reward_scheme.self_improvement_bonus == 0.1
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Self-Rewarding Scheme'
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)