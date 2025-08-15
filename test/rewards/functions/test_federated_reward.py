"""
测试联邦奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.federated_reward import FederatedReward


class TestFederatedReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = FederatedReward(
            num_agents=3,
            aggregation_method='average',
            privacy_weight=0.1,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.num_agents == 3
        assert self.reward_scheme.aggregation_method == 'average'
        assert self.reward_scheme.privacy_weight == 0.1
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Federated Learning Reward'
    
    def test_reward_aggregation(self):
        """测试奖励聚合"""
        agent_rewards = [0.1, 0.2, -0.05]
        
        # 测试平均聚合
        avg_reward = self.reward_scheme._aggregate_rewards(agent_rewards, 'average')
        expected_avg = np.mean(agent_rewards)
        assert abs(avg_reward - expected_avg) < 1e-6
        
        # 测试加权聚合
        weighted_reward = self.reward_scheme._aggregate_rewards(agent_rewards, 'weighted')
        assert isinstance(weighted_reward, (int, float))
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)