"""
测试制度感知奖励函数模块
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.environment.rewards.regime_aware import RegimeAwareReward


class TestRegimeAwareReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = RegimeAwareReward(
            regime_window=20,
            volatility_threshold=0.02,
            initial_balance=10000.0
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.regime_window == 20
        assert self.reward_scheme.volatility_threshold == 0.02
        assert self.reward_scheme.initial_balance == 10000.0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        assert isinstance(info, dict)
        assert info['name'] == 'Regime-Aware Reward'
    
    def test_regime_detection(self):
        """测试制度检测"""
        # 创建测试数据
        stable_returns = [0.01] * 20  # 稳定制度
        volatile_returns = [0.1, -0.08, 0.15, -0.12] * 5  # 波动制度
        
        stable_regime = self.reward_scheme._detect_regime(stable_returns)
        volatile_regime = self.reward_scheme._detect_regime(volatile_returns)
        
        assert isinstance(stable_regime, str)
        assert isinstance(volatile_regime, str)
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)