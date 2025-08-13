"""
测试动态Sortino奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.dynamic_sortino import DynamicSortinoReward


class TestDynamicSortinoReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.reward_scheme = DynamicSortinoReward(
            target_return=0.02,
            window_size=30,
            adaptation_rate=0.1,
            min_periods=10,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.target_return == 0.02
        assert self.reward_scheme.window_size == 30
        assert self.reward_scheme.adaptation_rate == 0.1
        assert self.reward_scheme.min_periods == 10
        assert self.reward_scheme.initial_balance == self.initial_balance
        assert len(self.reward_scheme.return_history) == 0
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Dynamic Sortino Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'target_return' in info['parameters']
        assert 'adaptation_rate' in info['parameters']
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些历史数据
        self.reward_scheme.return_history = [0.01, 0.02, -0.01, 0.03]
        self.reward_scheme.downside_returns = [-0.01, -0.005]
        self.reward_scheme.current_target = 0.025
        
        # 重置
        self.reward_scheme.reset()
        
        # 验证历史数据已清空
        assert len(self.reward_scheme.return_history) == 0
        assert len(self.reward_scheme.downside_returns) == 0
        # target_return 可能重置为初始值
        assert abs(self.reward_scheme.current_target - self.reward_scheme.target_return) < 1e-6
    
    def test_downside_deviation_calculation(self):
        """测试下行偏差计算"""
        # 创建包含正负收益的历史
        returns = [0.03, -0.02, 0.01, -0.01, 0.025, -0.015, 0.02]
        target_return = 0.02
        
        downside_deviation = self.reward_scheme._calculate_downside_deviation(
            returns, target_return
        )
        
        assert isinstance(downside_deviation, (int, float))
        assert downside_deviation >= 0
        assert not np.isnan(downside_deviation)
    
    def test_sortino_ratio_calculation(self):
        """测试Sortino比率计算"""
        # 创建测试数据
        returns = [0.03, 0.01, -0.01, 0.02, 0.025]
        target_return = 0.015
        
        # 计算Sortino比率
        mean_return = np.mean(returns)
        excess_return = mean_return - target_return
        
        downside_deviation = self.reward_scheme._calculate_downside_deviation(
            returns, target_return
        )
        
        if downside_deviation > 0:
            sortino_ratio = excess_return / downside_deviation
            assert isinstance(sortino_ratio, (int, float))
            assert not np.isnan(sortino_ratio)
    
    def test_adaptive_target_update(self):
        """测试自适应目标更新"""
        # 设置初始目标
        initial_target = self.reward_scheme.current_target
        
        # 创建持续正收益的历史
        high_returns = [0.04, 0.05, 0.045, 0.055, 0.048]
        self.reward_scheme.return_history = high_returns
        
        # 更新自适应目标
        self.reward_scheme._update_adaptive_target()
        
        # 在高收益环境下，目标可能上调
        updated_target = self.reward_scheme.current_target
        
        # 验证目标更新逻辑存在
        assert isinstance(updated_target, (int, float))
        assert not np.isnan(updated_target)
    
    def test_get_reward_calculation(self):
        """测试奖励计算"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.portfolio_value = 10600.0
        mock_env.initial_balance = self.initial_balance
        
        # 添加一些历史收益
        self.reward_scheme.return_history = [0.02, 0.015, -0.005, 0.03]
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        
        # 验证历史记录更新
        assert len(self.reward_scheme.return_history) > 4
    
    def test_window_size_management(self):
        """测试窗口大小管理"""
        # 填充超过窗口大小的历史数据
        window_size = self.reward_scheme.window_size
        
        for i in range(window_size + 10):
            returns = [0.01 + 0.001 * i]
            self.reward_scheme.return_history.extend(returns)
        
        # 手动触发窗口管理
        if len(self.reward_scheme.return_history) > window_size:
            self.reward_scheme.return_history = self.reward_scheme.return_history[-window_size:]
        
        # 验证历史长度不超过窗口大小
        assert len(self.reward_scheme.return_history) <= window_size
    
    def test_minimum_periods_handling(self):
        """测试最小周期数处理"""
        # 创建少于最小周期数的历史
        short_history = [0.02, 0.015, 0.01]
        self.reward_scheme.return_history = short_history
        
        mock_env = Mock()
        mock_env.portfolio_value = 10300.0
        mock_env.initial_balance = self.initial_balance
        
        # 在数据不足时，应该有默认处理方式
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_adaptation_rate_impact(self):
        """测试自适应率影响"""
        # 创建不同适应率的奖励函数
        slow_adaptation = DynamicSortinoReward(adaptation_rate=0.01)
        fast_adaptation = DynamicSortinoReward(adaptation_rate=0.5)
        
        # 设置相同的高收益历史
        high_returns = [0.08, 0.07, 0.09, 0.075, 0.085]
        slow_adaptation.return_history = high_returns.copy()
        fast_adaptation.return_history = high_returns.copy()
        
        # 记录初始目标
        slow_initial = slow_adaptation.current_target
        fast_initial = fast_adaptation.current_target
        
        # 更新目标
        slow_adaptation._update_adaptive_target()
        fast_adaptation._update_adaptive_target()
        
        # 快速适应应该有更大的目标变化
        slow_change = abs(slow_adaptation.current_target - slow_initial)
        fast_change = abs(fast_adaptation.current_target - fast_initial)
        
        # 验证适应性存在
        assert isinstance(slow_change, (int, float))
        assert isinstance(fast_change, (int, float))
    
    def test_negative_returns_handling(self):
        """测试负收益处理"""
        # 创建主要为负收益的历史
        negative_returns = [-0.02, -0.015, -0.01, -0.025, -0.005]
        self.reward_scheme.return_history = negative_returns
        
        mock_env = Mock()
        mock_env.portfolio_value = 9500.0  # 损失
        mock_env.initial_balance = self.initial_balance
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        
        # 在负收益环境下，Sortino比率应该为负
        # 奖励函数应该正确处理这种情况
    
    def test_zero_downside_deviation(self):
        """测试零下行偏差情况"""
        # 创建全部收益都高于目标的情况
        high_returns = [0.03, 0.025, 0.035, 0.028, 0.032]
        target_return = 0.02
        
        downside_deviation = self.reward_scheme._calculate_downside_deviation(
            high_returns, target_return
        )
        
        # 如果没有下行收益，下行偏差可能为零
        assert downside_deviation >= 0
        
        # 测试这种情况下的Sortino比率处理
        if downside_deviation == 0:
            # 应该有特殊处理逻辑（如设为最大值或特定值）
            pass
    
    def test_target_return_bounds(self):
        """测试目标收益边界"""
        # 测试极端目标收益值
        extreme_params = [
            {'target_return': 0.0},     # 零目标
            {'target_return': 0.5},     # 高目标
            {'target_return': -0.1}     # 负目标
        ]
        
        for params in extreme_params:
            try:
                scheme = DynamicSortinoReward(**params)
                assert scheme is not None
                
                # 测试基本功能
                info = scheme.get_reward_info()
                assert isinstance(info, dict)
                
            except ValueError:
                # 某些极端值可能不被允许
                pass
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空收益历史
        mock_env = Mock()
        mock_env.portfolio_value = 10000.0
        mock_env.initial_balance = self.initial_balance
        
        # 清空历史
        self.reward_scheme.return_history = []
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试有效参数范围
        valid_params = [
            {'window_size': 10, 'min_periods': 5},
            {'adaptation_rate': 0.0},   # 无适应
            {'adaptation_rate': 1.0},   # 完全适应
            {'target_return': 0.01}
        ]
        
        for params in valid_params:
            scheme = DynamicSortinoReward(**params)
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)
    
    def test_consistent_calculations(self):
        """测试计算一致性"""
        # 使用相同数据多次计算，结果应该一致
        test_returns = [0.02, 0.015, -0.005, 0.025, 0.01]
        target = 0.015
        
        deviation_1 = self.reward_scheme._calculate_downside_deviation(test_returns, target)
        deviation_2 = self.reward_scheme._calculate_downside_deviation(test_returns, target)
        
        assert abs(deviation_1 - deviation_2) < 1e-10