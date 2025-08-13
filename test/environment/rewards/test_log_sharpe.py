"""
测试对数夏普奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.log_sharpe import LogSharpeReward


class TestLogSharpeReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = LogSharpeReward(
            risk_free_rate=0.02,
            window_size=50,
            smoothing_factor=0.1,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.risk_free_rate == 0.02
        assert self.reward_scheme.window_size == 50
        assert self.reward_scheme.smoothing_factor == 0.1
        assert self.reward_scheme.initial_balance == self.initial_balance
        assert len(self.reward_scheme.log_returns) == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Log Sharpe Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'smoothing_factor' in info['parameters']
    
    def test_log_return_calculation(self):
        """测试对数收益率计算"""
        prev_value = 10000.0
        curr_value = 11000.0
        
        log_return = self.reward_scheme._calculate_log_return(prev_value, curr_value)
        
        assert isinstance(log_return, (int, float))
        assert not np.isnan(log_return)
        
        # 验证对数收益率公式
        expected_log_return = np.log(curr_value / prev_value)
        assert abs(log_return - expected_log_return) < 1e-10
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        mock_env = Mock()
        mock_env.portfolio_value = 10800.0
        mock_env.initial_balance = self.initial_balance
        
        # 添加一些历史对数收益
        self.reward_scheme.log_returns = [0.02, 0.015, -0.005, 0.03, 0.01]
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_smoothing_effect(self):
        """测试平滑效果"""
        # 创建不同平滑因子的奖励函数
        no_smoothing = LogSharpeReward(smoothing_factor=0.0)
        heavy_smoothing = LogSharpeReward(smoothing_factor=0.9)
        
        # 添加相同的历史数据
        test_returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        no_smoothing.log_returns = test_returns.copy()
        heavy_smoothing.log_returns = test_returns.copy()
        
        # 计算平滑后的夏普比率
        no_smooth_sharpe = no_smoothing._calculate_smoothed_sharpe()
        heavy_smooth_sharpe = heavy_smoothing._calculate_smoothed_sharpe()
        
        assert isinstance(no_smooth_sharpe, (int, float))
        assert isinstance(heavy_smooth_sharpe, (int, float))
    
    def test_parameter_validation(self):
        """测试参数验证"""
        valid_schemes = [
            LogSharpeReward(smoothing_factor=0.0),
            LogSharpeReward(smoothing_factor=1.0),
            LogSharpeReward(window_size=10)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)