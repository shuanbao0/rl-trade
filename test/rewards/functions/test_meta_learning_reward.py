"""
测试元学习奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.meta_learning_reward import MetaLearningReward


class TestMetaLearningReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = MetaLearningReward(
            adaptation_rate=0.1,
            memory_size=100,
            meta_learning_weight=0.2,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.adaptation_rate == 0.1
        assert self.reward_scheme.memory_size == 100
        assert self.reward_scheme.meta_learning_weight == 0.2
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Meta-Learning Reward'
    
    def test_adaptation_mechanism(self):
        """测试适应机制"""
        # 模拟任务经验
        task_experiences = [
            {'task_id': 1, 'reward': 0.1, 'context': [1, 2, 3]},
            {'task_id': 2, 'reward': 0.2, 'context': [4, 5, 6]}
        ]
        
        adapted_strategy = self.reward_scheme._adapt_strategy(task_experiences)
        assert isinstance(adapted_strategy, dict)
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)