"""
测试多模态奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.multimodal_reward import MultimodalReward


class TestMultimodalReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = MultimodalReward(
            modality_weights={'price': 0.4, 'news': 0.3, 'volume': 0.3},
            fusion_method='attention',
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        expected_weights = {'price': 0.4, 'news': 0.3, 'volume': 0.3}
        assert self.reward_scheme.modality_weights == expected_weights
        assert self.reward_scheme.fusion_method == 'attention'
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Multimodal Reward'
    
    def test_modality_fusion(self):
        """测试模态融合"""
        mock_modalities = {
            'price': np.array([0.1, 0.2, 0.3]),
            'news': np.array([0.05, 0.15, 0.25]),
            'volume': np.array([0.08, 0.18, 0.28])
        }
        
        fused_features = self.reward_scheme._fuse_modalities(mock_modalities)
        assert isinstance(fused_features, np.ndarray)
        assert len(fused_features) > 0
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)