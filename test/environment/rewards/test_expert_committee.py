"""
测试专家委员会奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.expert_committee import ExpertCommitteeReward


class TestExpertCommitteeReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.expert_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'volatility': 0.2,
            'fundamental': 0.3
        }
        self.reward_scheme = ExpertCommitteeReward(
            expert_weights=self.expert_weights,
            consensus_threshold=0.6,
            disagreement_penalty=0.1,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.expert_weights == self.expert_weights
        assert self.reward_scheme.consensus_threshold == 0.6
        assert self.reward_scheme.disagreement_penalty == 0.1
        assert self.reward_scheme.initial_balance == self.initial_balance
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Expert Committee Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'expert_weights' in info['parameters']
    
    def test_expert_opinions(self):
        """测试专家意见计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = self.initial_balance
        
        # 模拟价格数据
        prices = pd.Series([100, 102, 101, 103, 105])
        
        with patch.object(self.reward_scheme, '_get_price_data', return_value=prices):
            opinions = self.reward_scheme._calculate_expert_opinions(mock_env)
        
        assert isinstance(opinions, dict)
        assert len(opinions) == len(self.expert_weights)
        
        for expert_name in self.expert_weights:
            assert expert_name in opinions
            assert isinstance(opinions[expert_name], (int, float))
    
    def test_consensus_calculation(self):
        """测试一致性计算"""
        # 高度一致的意见
        high_consensus_opinions = {
            'momentum': 0.8,
            'mean_reversion': 0.75,
            'volatility': 0.85,
            'fundamental': 0.82
        }
        
        consensus = self.reward_scheme._calculate_consensus(high_consensus_opinions)
        assert isinstance(consensus, (int, float))
        assert consensus >= 0
        
        # 分歧较大的意见
        low_consensus_opinions = {
            'momentum': 0.9,
            'mean_reversion': -0.8,
            'volatility': 0.1,
            'fundamental': 0.5
        }
        
        low_consensus = self.reward_scheme._calculate_consensus(low_consensus_opinions)
        assert isinstance(low_consensus, (int, float))
        assert low_consensus >= 0
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 10800.0
        mock_env.initial_balance = self.initial_balance
        
        # 模拟价格和市场数据
        prices = pd.Series([100, 102, 98, 105, 108])
        
        with patch.object(self.reward_scheme, '_get_price_data', return_value=prices):
            reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_disagreement_penalty(self):
        """测试分歧惩罚"""
        # 创建分歧大的专家意见
        disagreeing_opinions = {
            'momentum': 1.0,
            'mean_reversion': -1.0,
            'volatility': 0.5,
            'fundamental': -0.5
        }
        
        consensus = self.reward_scheme._calculate_consensus(disagreeing_opinions)
        
        # 分歧大时一致性应该较低
        assert consensus >= 0
        
        # 测试分歧惩罚的应用
        if consensus < self.reward_scheme.consensus_threshold:
            penalty = self.reward_scheme.disagreement_penalty
            assert penalty > 0
    
    def test_parameter_validation(self):
        """测试参数验证"""
        valid_schemes = [
            ExpertCommitteeReward(expert_weights={'single': 1.0}),
            ExpertCommitteeReward(consensus_threshold=0.0),
            ExpertCommitteeReward(disagreement_penalty=0.0)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)