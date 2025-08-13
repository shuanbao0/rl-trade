"""
测试盈亏奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.profit_loss import ProfitLossReward


class TestProfitLossReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = ProfitLossReward(
            min_trade_threshold=0.001,
            profit_bonus=2.0,
            loss_penalty=1.5,
            consecutive_loss_penalty=0.5,
            win_rate_bonus=0.1,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.min_trade_threshold == 0.001
        assert self.reward_scheme.profit_bonus == 2.0
        assert self.reward_scheme.loss_penalty == 1.5
        assert self.reward_scheme.consecutive_loss_penalty == 0.5
        assert self.reward_scheme.win_rate_bonus == 0.1
        assert self.reward_scheme.initial_balance == self.initial_balance
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Profit & Loss Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'profit_bonus' in info['parameters']
        assert 'loss_penalty' in info['parameters']
    
    def test_trade_classification(self):
        """测试交易分类"""
        # 盈利交易
        profit_return = 0.05
        is_profit = self.reward_scheme._is_profitable_trade(profit_return)
        assert is_profit == True
        
        # 亏损交易
        loss_return = -0.03
        is_loss = self.reward_scheme._is_profitable_trade(loss_return)
        assert is_loss == False
        
        # 微小变化（低于阈值）
        small_return = 0.0005
        is_small = self.reward_scheme._is_significant_trade(small_return)
        assert is_small == False
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 11200.0
        mock_env.initial_balance = self.initial_balance
        
        # 模拟交易历史
        mock_env.trade_history = [0.03, -0.01, 0.02, 0.05]
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_consecutive_losses_tracking(self):
        """测试连续亏损跟踪"""
        # 创建连续亏损序列
        loss_sequence = [-0.02, -0.015, -0.01, 0.05, -0.008]
        
        consecutive_losses = self.reward_scheme._count_consecutive_losses(loss_sequence)
        
        assert isinstance(consecutive_losses, int)
        assert consecutive_losses >= 0
    
    def test_win_rate_calculation(self):
        """测试胜率计算"""
        # 创建混合交易历史
        trade_results = [0.03, -0.01, 0.02, -0.025, 0.04, 0.01]
        
        win_rate = self.reward_scheme._calculate_win_rate(trade_results)
        
        assert isinstance(win_rate, (int, float))
        assert 0 <= win_rate <= 1
    
    def test_parameter_validation(self):
        """测试参数验证"""
        valid_schemes = [
            ProfitLossReward(profit_bonus=0.0),
            ProfitLossReward(loss_penalty=0.0),
            ProfitLossReward(min_trade_threshold=0.0)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)