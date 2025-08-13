"""
测试不确定性感知奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.uncertainty_aware import UncertaintyAwareReward


class TestUncertaintyAwareReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = UncertaintyAwareReward(
            uncertainty_weight=0.3,
            confidence_threshold=0.7,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.uncertainty_weight == 0.3
        assert self.reward_scheme.confidence_threshold == 0.7
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Uncertainty-Aware Reward'
    
    def test_uncertainty_estimation(self):
        """测试不确定性估计"""
        mock_predictions = np.array([0.1, 0.8, 0.1])  # 高置信度
        low_uncertainty = self.reward_scheme._estimate_uncertainty(mock_predictions)
        
        mock_predictions_uncertain = np.array([0.4, 0.3, 0.3])  # 低置信度
        high_uncertainty = self.reward_scheme._estimate_uncertainty(mock_predictions_uncertain)
        
        assert isinstance(low_uncertainty, (int, float))
        assert isinstance(high_uncertainty, (int, float))
        assert high_uncertainty > low_uncertainty
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)