"""
测试LLM引导奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.llm_guided import LLMGuidedReward


class TestLLMGuidedReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = LLMGuidedReward(
            llm_weight=0.3,
            guidance_threshold=0.5,
            learning_rate=0.01,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.llm_weight == 0.3
        assert self.reward_scheme.guidance_threshold == 0.5
        assert self.reward_scheme.learning_rate == 0.01
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'LLM-Guided Reward'
    
    def test_llm_guidance_integration(self):
        """测试LLM引导集成"""
        mock_guidance = {
            'action_score': 0.8,
            'confidence': 0.9,
            'reasoning': 'Good momentum signal'
        }
        
        guidance_reward = self.reward_scheme._integrate_llm_guidance(mock_guidance)
        assert isinstance(guidance_reward, (int, float))
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)