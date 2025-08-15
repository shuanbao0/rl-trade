"""
测试人类反馈强化学习奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.rlhf_reward import RLHFReward


class TestRLHFReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = RLHFReward(
            human_feedback_weight=0.4,
            preference_model_weight=0.6,
            feedback_buffer_size=1000,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.human_feedback_weight == 0.4
        assert self.reward_scheme.preference_model_weight == 0.6
        assert self.reward_scheme.feedback_buffer_size == 1000
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'RLHF Reward'
    
    def test_human_feedback_integration(self):
        """测试人类反馈集成"""
        mock_feedback = {
            'action_id': 1,
            'human_rating': 0.8,
            'preference_score': 0.7,
            'confidence': 0.9
        }
        
        integrated_reward = self.reward_scheme._integrate_human_feedback(mock_feedback)
        assert isinstance(integrated_reward, (int, float))
    
    def test_preference_model_update(self):
        """测试偏好模型更新"""
        feedback_pairs = [
            {'action_a': [0.1, 0.2], 'action_b': [0.3, 0.4], 'preference': 'b'},
            {'action_a': [0.5, 0.6], 'action_b': [0.2, 0.3], 'preference': 'a'}
        ]
        
        model_updated = self.reward_scheme._update_preference_model(feedback_pairs)
        assert isinstance(model_updated, bool)
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)