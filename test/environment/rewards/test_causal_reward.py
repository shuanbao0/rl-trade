"""
测试因果奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.causal_reward import CausalReward


class TestCausalReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = CausalReward(
            causal_strength=0.5,
            intervention_threshold=0.02,
            lookback_window=20,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.causal_strength == 0.5
        assert self.reward_scheme.intervention_threshold == 0.02
        assert self.reward_scheme.lookback_window == 20
        assert self.reward_scheme.initial_balance == self.initial_balance
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Causal Reward'
        assert 'description' in info
        assert 'parameters' in info
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些状态
        self.reward_scheme.causal_history = [1, 2, 3]
        self.reward_scheme.intervention_count = 5
        
        # 重置
        self.reward_scheme.reset()
        
        # 验证状态已清空
        assert len(self.reward_scheme.causal_history) == 0
        assert self.reward_scheme.intervention_count == 0
    
    def test_compute_causal_effect(self):
        """测试因果效应计算"""
        # 创建测试数据
        prices = np.array([100, 102, 101, 103, 105])
        action = 0.5
        
        effect = self.reward_scheme._compute_causal_effect(prices, action)
        
        assert isinstance(effect, float)
        assert not np.isnan(effect)
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        # 创建模拟环境状态
        mock_env = Mock()
        mock_env.portfolio_value = 10500.0
        mock_env.initial_balance = self.initial_balance
        mock_env.action_history = [0.1, -0.2, 0.3]
        
        # 创建价格历史
        prices = pd.Series([100, 102, 101, 103, 105], name='Close')
        
        with patch.object(self.reward_scheme, '_get_price_series', return_value=prices):
            reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_intervention_detection(self):
        """测试干预检测"""
        # 创建强动作序列（应触发干预）
        strong_actions = [0.8, -0.9, 0.7]
        weak_actions = [0.01, -0.005, 0.015]
        
        # 测试强动作
        intervention_strong = any(
            abs(action) > self.reward_scheme.intervention_threshold 
            for action in strong_actions
        )
        assert intervention_strong
        
        # 测试弱动作
        intervention_weak = any(
            abs(action) > self.reward_scheme.intervention_threshold 
            for action in weak_actions
        )
        assert not intervention_weak
    
    def test_causal_strength_impact(self):
        """测试因果强度对奖励的影响"""
        # 创建两个不同因果强度的奖励函数
        weak_causal = CausalReward(causal_strength=0.1)
        strong_causal = CausalReward(causal_strength=0.9)
        
        # 创建相同的测试环境
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        mock_env.action_history = [0.5, -0.3, 0.4]
        
        prices = pd.Series([100, 102, 98, 105, 108])
        
        with patch.object(weak_causal, '_get_price_series', return_value=prices), \
             patch.object(strong_causal, '_get_price_series', return_value=prices):
            
            weak_reward = weak_causal.get_reward(mock_env)
            strong_reward = strong_causal.get_reward(mock_env)
            
            # 验证奖励都是有效数值
            assert isinstance(weak_reward, (int, float))
            assert isinstance(strong_reward, (int, float))
            assert not np.isnan(weak_reward)
            assert not np.isnan(strong_reward)
    
    def test_lookback_window_handling(self):
        """测试回看窗口处理"""
        # 测试短历史（少于回看窗口）
        short_prices = pd.Series([100, 102, 101])
        
        mock_env = Mock()
        mock_env.portfolio_value = 10200.0
        mock_env.initial_balance = 10000.0
        mock_env.action_history = [0.1]
        
        with patch.object(self.reward_scheme, '_get_price_series', return_value=short_prices):
            reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_edge_cases(self):
        """测试边界情况"""
        mock_env = Mock()
        mock_env.portfolio_value = 10000.0  # 无变化
        mock_env.initial_balance = 10000.0
        mock_env.action_history = []  # 空历史
        
        # 测试空价格序列
        empty_prices = pd.Series([], dtype=float)
        
        with patch.object(self.reward_scheme, '_get_price_series', return_value=empty_prices):
            reward = self.reward_scheme.get_reward(mock_env)
        
        # 应该有默认处理机制
        assert isinstance(reward, (int, float))
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试有效参数范围
        valid_schemes = [
            CausalReward(causal_strength=0.0),
            CausalReward(causal_strength=1.0),
            CausalReward(intervention_threshold=0.001),
            CausalReward(lookback_window=5)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)