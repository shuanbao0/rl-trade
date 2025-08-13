"""
测试收益回撤奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.return_drawdown import ReturnDrawdownReward


class TestReturnDrawdownReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = ReturnDrawdownReward(
            return_weight=0.7,
            drawdown_weight=0.3,
            max_drawdown_threshold=0.2,
            recovery_bonus=0.05,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.return_weight == 0.7
        assert self.reward_scheme.drawdown_weight == 0.3
        assert self.reward_scheme.max_drawdown_threshold == 0.2
        assert self.reward_scheme.recovery_bonus == 0.05
        assert self.reward_scheme.initial_balance == self.initial_balance
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Return-Drawdown Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'return_weight' in info['parameters']
    
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        # 创建价值历史（包含回撤）
        value_history = [10000, 11000, 10500, 9500, 10200, 12000]
        
        max_drawdown = self.reward_scheme._calculate_max_drawdown(value_history)
        
        assert isinstance(max_drawdown, (int, float))
        assert max_drawdown <= 0  # 回撤应该是负值或零
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11500.0
        mock_env.initial_balance = self.initial_balance
        mock_env.value_history = [10000, 10500, 11000, 10800, 11500]
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_recovery_detection(self):
        """测试恢复检测"""
        # 创建包含恢复的价值历史
        recovery_history = [10000, 11000, 9000, 9500, 10500, 11200]
        
        is_recovering = self.reward_scheme._detect_recovery(recovery_history)
        
        assert isinstance(is_recovering, bool)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        valid_schemes = [
            ReturnDrawdownReward(return_weight=1.0, drawdown_weight=0.0),
            ReturnDrawdownReward(max_drawdown_threshold=0.5),
            ReturnDrawdownReward(recovery_bonus=0.0)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)