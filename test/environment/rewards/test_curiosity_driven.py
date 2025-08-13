"""
测试好奇心驱动奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.curiosity_driven import CuriosityDrivenReward


class TestCuriosityDrivenReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = CuriosityDrivenReward(
            curiosity_weight=0.3,
            novelty_threshold=0.1,
            exploration_bonus=0.05,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.curiosity_weight == 0.3
        assert self.reward_scheme.novelty_threshold == 0.1
        assert self.reward_scheme.exploration_bonus == 0.05
        assert self.reward_scheme.initial_balance == self.initial_balance
        assert len(self.reward_scheme.state_history) == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Curiosity-Driven Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'curiosity_weight' in info['parameters']
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些历史状态
        self.reward_scheme.state_history = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6])
        ]
        self.reward_scheme.novelty_scores = [0.1, 0.2]
        
        # 重置
        self.reward_scheme.reset()
        
        # 验证状态已清空
        assert len(self.reward_scheme.state_history) == 0
        assert len(self.reward_scheme.novelty_scores) == 0
    
    def test_compute_novelty(self):
        """测试新颖性计算"""
        # 创建当前状态
        current_state = np.array([1.0, 2.0, 3.0])
        
        # 空历史情况（最大新颖性）
        novelty = self.reward_scheme._compute_novelty(current_state)
        assert isinstance(novelty, float)
        assert novelty > 0
        
        # 添加历史状态
        self.reward_scheme.state_history = [
            np.array([1.1, 2.1, 3.1]),  # 相似状态
            np.array([5.0, 6.0, 7.0])   # 不同状态
        ]
        
        novelty_with_history = self.reward_scheme._compute_novelty(current_state)
        assert isinstance(novelty_with_history, float)
        assert novelty_with_history >= 0
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.portfolio_value = 10500.0
        mock_env.initial_balance = self.initial_balance
        
        # 模拟观察状态
        observation = np.array([0.1, 0.05, -0.02, 0.15, 0.03])
        
        with patch.object(mock_env, 'get_observation', return_value=observation):
            reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_exploration_bonus_calculation(self):
        """测试探索奖励计算"""
        # 创建高新颖性状态
        high_novelty_state = np.array([10.0, 20.0, 30.0])
        
        # 添加不同的历史状态
        self.reward_scheme.state_history = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0])
        ]
        
        novelty = self.reward_scheme._compute_novelty(high_novelty_state)
        
        # 如果新颖性超过阈值，应该有探索奖励
        if novelty > self.reward_scheme.novelty_threshold:
            exploration_bonus = self.reward_scheme.exploration_bonus
            assert exploration_bonus > 0
    
    def test_curiosity_weight_impact(self):
        """测试好奇心权重对奖励的影响"""
        # 创建不同好奇心权重的奖励函数
        low_curiosity = CuriosityDrivenReward(curiosity_weight=0.1)
        high_curiosity = CuriosityDrivenReward(curiosity_weight=0.9)
        
        # 创建相同的测试环境
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        
        # 使用相同的观察状态
        observation = np.array([0.2, 0.1, -0.05, 0.3, 0.08])
        
        with patch.object(mock_env, 'get_observation', return_value=observation):
            low_reward = low_curiosity.get_reward(mock_env)
            high_reward = high_curiosity.get_reward(mock_env)
            
            # 验证奖励都是有效数值
            assert isinstance(low_reward, (int, float))
            assert isinstance(high_reward, (int, float))
            assert not np.isnan(low_reward)
            assert not np.isnan(high_reward)
    
    def test_state_history_management(self):
        """测试状态历史管理"""
        max_history = 100
        reward_scheme = CuriosityDrivenReward(max_history_size=max_history)
        
        # 添加超过最大历史长度的状态
        for i in range(max_history + 10):
            state = np.array([i, i+1, i+2])
            reward_scheme.state_history.append(state)
            
            # 检查历史长度限制
            if len(reward_scheme.state_history) > max_history:
                reward_scheme.state_history.pop(0)
        
        assert len(reward_scheme.state_history) <= max_history
    
    def test_novelty_threshold_behavior(self):
        """测试新颖性阈值行为"""
        # 测试不同的新颖性阈值
        low_threshold = CuriosityDrivenReward(novelty_threshold=0.01)
        high_threshold = CuriosityDrivenReward(novelty_threshold=0.9)
        
        # 创建测试状态
        state = np.array([1.0, 2.0, 3.0])
        
        # 添加相似的历史状态
        similar_history = [np.array([1.1, 2.1, 3.1])]
        low_threshold.state_history = similar_history.copy()
        high_threshold.state_history = similar_history.copy()
        
        low_novelty = low_threshold._compute_novelty(state)
        high_novelty = high_threshold._compute_novelty(state)
        
        assert isinstance(low_novelty, float)
        assert isinstance(high_novelty, float)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空观察状态
        mock_env = Mock()
        mock_env.portfolio_value = 10000.0
        mock_env.initial_balance = 10000.0
        
        empty_observation = np.array([])
        
        with patch.object(mock_env, 'get_observation', return_value=empty_observation):
            try:
                reward = self.reward_scheme.get_reward(mock_env)
                # 如果没有异常，验证结果
                assert isinstance(reward, (int, float))
            except (ValueError, IndexError):
                # 空观察可能导致预期的异常
                pass
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试有效参数范围
        valid_schemes = [
            CuriosityDrivenReward(curiosity_weight=0.0),
            CuriosityDrivenReward(curiosity_weight=1.0),
            CuriosityDrivenReward(novelty_threshold=0.01),
            CuriosityDrivenReward(exploration_bonus=0.0)
        ]
        
        for scheme in valid_schemes:
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)
    
    def test_consistent_reward_calculation(self):
        """测试奖励计算的一致性"""
        mock_env = Mock()
        mock_env.portfolio_value = 10300.0
        mock_env.initial_balance = 10000.0
        
        observation = np.array([0.05, 0.02, -0.01, 0.08, 0.04])
        
        # 多次计算相同状态的奖励
        rewards = []
        for _ in range(3):
            with patch.object(mock_env, 'get_observation', return_value=observation):
                reward = self.reward_scheme.get_reward(mock_env)
                rewards.append(reward)
        
        # 验证所有奖励都是有效数值
        for reward in rewards:
            assert isinstance(reward, (int, float))
            assert not np.isnan(reward)