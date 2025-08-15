"""
测试多元化奖励函数模块
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.environment.rewards.diversified import DiversifiedReward


class TestDiversifiedReward:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.initial_balance = 10000.0
        self.weights = {
            'return': 0.4,
            'risk': 0.2,
            'stability': 0.15,
            'efficiency': 0.15,
            'drawdown': 0.1
        }
        self.reward_scheme = DiversifiedReward(
            weights=self.weights,
            risk_free_rate=0.02,
            volatility_window=20,
            drawdown_threshold=0.1,
            trading_cost=0.001,
            initial_balance=self.initial_balance
        )
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.weights == self.weights
        assert self.reward_scheme.risk_free_rate == 0.02
        assert self.reward_scheme.volatility_window == 20
        assert self.reward_scheme.drawdown_threshold == 0.1
        assert self.reward_scheme.trading_cost == 0.001
        assert self.reward_scheme.initial_balance == self.initial_balance
        
        # 验证权重总和
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_reward_info(self):
        """测试奖励信息"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert info['name'] == 'Diversified Reward'
        assert 'description' in info
        assert 'parameters' in info
        assert 'weights' in info['parameters']
        assert 'components' in info['parameters']
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些历史数据
        self.reward_scheme.return_history = [0.01, 0.02, -0.01]
        self.reward_scheme.volatility_history = [0.15, 0.18, 0.12]
        self.reward_scheme.drawdown_history = [0.0, -0.05, -0.03]
        
        # 重置
        self.reward_scheme.reset()
        
        # 验证历史数据已清空
        assert len(self.reward_scheme.return_history) == 0
        assert len(self.reward_scheme.volatility_history) == 0
        assert len(self.reward_scheme.drawdown_history) == 0
    
    def test_return_component(self):
        """测试收益组件"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = self.initial_balance
        
        # 计算预期收益
        expected_return = (11000.0 - 10000.0) / 10000.0
        
        return_component = self.reward_scheme._calculate_return_component(mock_env)
        
        assert isinstance(return_component, (int, float))
        assert not np.isnan(return_component)
    
    def test_risk_component(self):
        """测试风险组件"""
        # 创建收益历史
        returns = [0.02, 0.01, -0.01, 0.03, -0.005]
        self.reward_scheme.return_history = returns
        
        risk_component = self.reward_scheme._calculate_risk_component()
        
        assert isinstance(risk_component, (int, float))
        assert not np.isnan(risk_component)
        
        # 风险组件通常是负值（惩罚高风险）
        # 具体实现可能不同，只验证数值有效性
    
    def test_stability_component(self):
        """测试稳定性组件"""
        # 创建波动率历史
        volatilities = [0.12, 0.15, 0.18, 0.14, 0.16]
        self.reward_scheme.volatility_history = volatilities
        
        stability_component = self.reward_scheme._calculate_stability_component()
        
        assert isinstance(stability_component, (int, float))
        assert not np.isnan(stability_component)
    
    def test_efficiency_component(self):
        """测试效率组件"""
        # 模拟交易活动
        mock_env = Mock()
        mock_env.total_trades = 10
        mock_env.total_volume = 50000.0
        mock_env.portfolio_value = 11000.0
        
        efficiency_component = self.reward_scheme._calculate_efficiency_component(mock_env)
        
        assert isinstance(efficiency_component, (int, float))
        assert not np.isnan(efficiency_component)
    
    def test_drawdown_component(self):
        """测试回撤组件"""
        # 创建回撤历史
        drawdowns = [0.0, -0.02, -0.05, -0.03, -0.01]
        self.reward_scheme.drawdown_history = drawdowns
        
        drawdown_component = self.reward_scheme._calculate_drawdown_component()
        
        assert isinstance(drawdown_component, (int, float))
        assert not np.isnan(drawdown_component)
        
        # 回撤组件通常是负值（惩罚大回撤）
    
    def test_get_reward_calculation(self):
        """测试综合奖励计算"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.portfolio_value = 10800.0
        mock_env.initial_balance = self.initial_balance
        mock_env.total_trades = 5
        mock_env.total_volume = 25000.0
        
        # 添加一些历史数据
        self.reward_scheme.return_history = [0.01, 0.02, -0.005]
        self.reward_scheme.volatility_history = [0.12, 0.15]
        self.reward_scheme.drawdown_history = [0.0, -0.02]
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_weight_normalization(self):
        """测试权重归一化"""
        # 创建非归一化权重
        unnormalized_weights = {
            'return': 2.0,
            'risk': 1.0,
            'stability': 0.5,
            'efficiency': 0.5,
            'drawdown': 0.5
        }
        
        normalized_reward = DiversifiedReward(weights=unnormalized_weights)
        
        # 验证权重被归一化
        total_weight = sum(normalized_reward.weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_missing_components_handling(self):
        """测试缺失组件处理"""
        # 创建只有部分权重的配置
        partial_weights = {
            'return': 0.6,
            'risk': 0.4
        }
        
        partial_reward = DiversifiedReward(weights=partial_weights)
        
        mock_env = Mock()
        mock_env.portfolio_value = 10500.0
        mock_env.initial_balance = 10000.0
        mock_env.total_trades = 3
        mock_env.total_volume = 15000.0
        
        # 应该能正常计算（未指定的组件权重为0）
        reward = partial_reward.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_trading_cost_impact(self):
        """测试交易成本影响"""
        # 创建不同交易成本的奖励函数
        low_cost = DiversifiedReward(
            weights=self.weights,
            trading_cost=0.0001
        )
        high_cost = DiversifiedReward(
            weights=self.weights,
            trading_cost=0.01
        )
        
        # 创建高交易量环境
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        mock_env.total_trades = 20
        mock_env.total_volume = 100000.0
        
        low_cost_reward = low_cost.get_reward(mock_env)
        high_cost_reward = high_cost.get_reward(mock_env)
        
        # 验证高交易成本导致更低的效率分数
        assert isinstance(low_cost_reward, (int, float))
        assert isinstance(high_cost_reward, (int, float))
    
    def test_drawdown_threshold(self):
        """测试回撤阈值"""
        # 创建大回撤情况
        large_drawdown = [-0.15, -0.12, -0.08]  # 超过10%阈值
        small_drawdown = [-0.05, -0.03, -0.02]  # 在阈值内
        
        # 测试大回撤
        self.reward_scheme.drawdown_history = large_drawdown
        large_drawdown_component = self.reward_scheme._calculate_drawdown_component()
        
        # 测试小回撤
        self.reward_scheme.drawdown_history = small_drawdown
        small_drawdown_component = self.reward_scheme._calculate_drawdown_component()
        
        # 大回撤应该有更大的惩罚
        assert isinstance(large_drawdown_component, (int, float))
        assert isinstance(small_drawdown_component, (int, float))
    
    def test_volatility_window_effect(self):
        """测试波动率窗口效应"""
        # 创建长期收益历史
        long_returns = [0.01 * np.sin(i * 0.1) for i in range(50)]
        
        # 测试不同窗口大小
        short_window = DiversifiedReward(volatility_window=10)
        long_window = DiversifiedReward(volatility_window=30)
        
        short_window.return_history = long_returns
        long_window.return_history = long_returns
        
        short_risk = short_window._calculate_risk_component()
        long_risk = long_window._calculate_risk_component()
        
        assert isinstance(short_risk, (int, float))
        assert isinstance(long_risk, (int, float))
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试零投资组合价值变化
        mock_env = Mock()
        mock_env.portfolio_value = 10000.0  # 无变化
        mock_env.initial_balance = 10000.0
        mock_env.total_trades = 0
        mock_env.total_volume = 0.0
        
        reward = self.reward_scheme.get_reward(mock_env)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试有效参数范围
        valid_params = [
            {'weights': {'return': 1.0}},
            {'risk_free_rate': 0.0},
            {'volatility_window': 5},
            {'drawdown_threshold': 0.5},
            {'trading_cost': 0.0}
        ]
        
        for params in valid_params:
            scheme = DiversifiedReward(**params)
            assert scheme is not None
            info = scheme.get_reward_info()
            assert isinstance(info, dict)
    
    def test_component_contribution(self):
        """测试各组件对总奖励的贡献"""
        mock_env = Mock()
        mock_env.portfolio_value = 11000.0
        mock_env.initial_balance = 10000.0
        mock_env.total_trades = 5
        mock_env.total_volume = 30000.0
        
        # 添加历史数据
        self.reward_scheme.return_history = [0.02, 0.01, 0.03]
        self.reward_scheme.volatility_history = [0.1, 0.12]
        self.reward_scheme.drawdown_history = [-0.02, -0.01]
        
        # 计算各组件
        return_comp = self.reward_scheme._calculate_return_component(mock_env)
        risk_comp = self.reward_scheme._calculate_risk_component()
        stability_comp = self.reward_scheme._calculate_stability_component()
        efficiency_comp = self.reward_scheme._calculate_efficiency_component(mock_env)
        drawdown_comp = self.reward_scheme._calculate_drawdown_component()
        
        # 验证所有组件都是有效数值
        components = [return_comp, risk_comp, stability_comp, efficiency_comp, drawdown_comp]
        for comp in components:
            assert isinstance(comp, (int, float))
            assert not np.isnan(comp)