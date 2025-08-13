"""
测试奖励函数基类
"""

import pytest
import numpy as np
from abc import ABCMeta
from src.environment.rewards.base_reward import BaseRewardScheme


class ConcreteReward(BaseRewardScheme):
    """具体的奖励函数实现用于测试"""
    
    def calculate_reward(self, portfolio_value: float, action: float, price: float, 
                        portfolio_info: dict, trade_info: dict, step: int, **kwargs) -> float:
        """简单的奖励实现"""
        self.update_history(portfolio_value)
        
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # 简单的收益率计算
        prev_value = self.portfolio_history[-2]
        if prev_value > 0:
            return_pct = (portfolio_value - prev_value) / prev_value * 100
        else:
            return_pct = 0.0
        
        reward = return_pct
        self.reward_history.append(reward)
        return reward
    
    def get_reward_info(self) -> dict:
        """返回奖励信息"""
        return {
            "name": "测试奖励函数",
            "description": "用于测试的简单奖励函数",
            "parameters": {
                "initial_balance": self.initial_balance
            }
        }


class TestBaseRewardScheme:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.reward_scheme = ConcreteReward(initial_balance=10000.0)
    
    def test_abstract_base_class(self):
        """测试抽象基类特性"""
        assert isinstance(BaseRewardScheme, ABCMeta)
        
        # 不能直接实例化抽象基类
        with pytest.raises(TypeError):
            BaseRewardScheme()
    
    def test_initialization(self):
        """测试初始化"""
        assert self.reward_scheme.initial_balance == 10000.0
        assert self.reward_scheme.previous_value is None
        assert len(self.reward_scheme.portfolio_history) == 0
        assert len(self.reward_scheme.reward_history) == 0
    
    def test_update_history(self):
        """测试历史更新"""
        # 更新几次历史
        values = [10000.0, 10500.0, 10200.0, 10800.0]
        
        for value in values:
            self.reward_scheme.update_history(value)
        
        assert len(self.reward_scheme.portfolio_history) == 4
        assert self.reward_scheme.portfolio_history == values
        assert self.reward_scheme.previous_value == values[-1]
    
    def test_get_reward_info_implementation(self):
        """测试奖励信息获取"""
        info = self.reward_scheme.get_reward_info()
        
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert info['name'] == "测试奖励函数"
    
    def test_calculate_reward_implementation(self):
        """测试奖励计算实现"""
        # 第一步
        reward1 = self.reward_scheme.calculate_reward(
            portfolio_value=10000.0,
            action=0.5,
            price=100.0,
            portfolio_info={},
            trade_info={},
            step=1
        )
        
        # 第一步应该返回0（没有历史比较）
        assert reward1 == 0.0
        
        # 第二步 - 增长
        reward2 = self.reward_scheme.calculate_reward(
            portfolio_value=11000.0,
            action=0.3,
            price=110.0,
            portfolio_info={},
            trade_info={},
            step=2
        )
        
        # 应该计算出正收益率
        expected_return = (11000.0 - 10000.0) / 10000.0 * 100  # 10%
        assert abs(reward2 - expected_return) < 0.01
    
    def test_history_tracking(self):
        """测试历史跟踪"""
        # 执行多步计算
        portfolio_values = [10000.0, 10200.0, 10100.0, 10300.0]
        
        for i, value in enumerate(portfolio_values):
            self.reward_scheme.calculate_reward(value, 0.1, 100.0, {}, {}, i + 1)
        
        # 验证投资组合历史
        assert len(self.reward_scheme.portfolio_history) == 4
        assert self.reward_scheme.portfolio_history == portfolio_values
        
        # 验证奖励历史（第一个应该是0）
        assert len(self.reward_scheme.reward_history) == 4
        assert self.reward_scheme.reward_history[0] == 0.0
        assert all(isinstance(r, (int, float)) for r in self.reward_scheme.reward_history)
    
    def test_backward_compatibility_reward_method(self):
        """测试向后兼容的reward方法"""
        from unittest.mock import MagicMock
        
        # 模拟TensorTrade环境
        mock_env = MagicMock()
        mock_env.portfolio = MagicMock()
        mock_env.portfolio.total_value = 10500.0
        
        # 模拟动作、价格等信息
        mock_env.action = 0.3
        mock_env.current_price = 105.0
        mock_env.current_step = 2
        
        # 设置初始历史
        self.reward_scheme.portfolio_history = [10000.0]
        
        # 调用兼容方法
        reward = self.reward_scheme.reward(mock_env)
        
        # 验证返回合理的奖励值
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试零值投资组合
        reward1 = self.reward_scheme.calculate_reward(0.0, 0.0, 0.0, {}, {}, 1)
        assert isinstance(reward1, (int, float))
        assert not np.isnan(reward1)
        
        # 测试负值投资组合
        reward2 = self.reward_scheme.calculate_reward(-1000.0, 0.0, 100.0, {}, {}, 1)
        assert isinstance(reward2, (int, float))
        assert not np.isnan(reward2)
        
        # 测试极大值投资组合
        reward3 = self.reward_scheme.calculate_reward(1e10, 1.0, 1000.0, {}, {}, 1)
        assert isinstance(reward3, (int, float))
        assert not np.isnan(reward3)
    
    def test_custom_initial_balance(self):
        """测试自定义初始余额"""
        custom_reward = ConcreteReward(initial_balance=50000.0)
        
        assert custom_reward.initial_balance == 50000.0
        assert custom_reward.previous_value is None
        assert len(custom_reward.portfolio_history) == 0
    
    def test_reward_consistency(self):
        """测试奖励一致性"""
        # 相同输入应该产生相同输出
        params = {
            'portfolio_value': 10500.0,
            'action': 0.2,
            'price': 105.0,
            'portfolio_info': {},
            'trade_info': {},
            'step': 2
        }
        
        # 设置相同的初始状态
        self.reward_scheme.portfolio_history = [10000.0]
        reward1 = self.reward_scheme.calculate_reward(**params)
        
        # 重置状态
        reward_scheme2 = ConcreteReward(initial_balance=10000.0)
        reward_scheme2.portfolio_history = [10000.0]
        reward2 = reward_scheme2.calculate_reward(**params)
        
        assert abs(reward1 - reward2) < 1e-10
    
    def test_portfolio_history_limit(self):
        """测试投资组合历史限制（如果实现了的话）"""
        # 添加大量历史数据
        for i in range(1000):
            self.reward_scheme.update_history(10000.0 + i)
        
        # 验证历史不会无限增长（如果有限制的话）
        assert len(self.reward_scheme.portfolio_history) <= 1000
        assert isinstance(self.reward_scheme.portfolio_history, list)
    
    def test_reward_parameters_validation(self):
        """测试奖励参数验证"""
        # 测试不同类型的参数
        reward = self.reward_scheme.calculate_reward(
            portfolio_value=10000.0,
            action=0.5,
            price=100.0,
            portfolio_info={'equity': 8000.0, 'cash': 2000.0},
            trade_info={'commission': 5.0, 'slippage': 0.1},
            step=1,
            custom_param=42
        )
        
        assert isinstance(reward, (int, float))
    
    def test_state_preservation(self):
        """测试状态保持"""
        # 执行一些操作
        self.reward_scheme.calculate_reward(10000.0, 0.1, 100.0, {}, {}, 1)
        self.reward_scheme.calculate_reward(10500.0, 0.2, 105.0, {}, {}, 2)
        
        # 保存状态
        saved_portfolio_history = self.reward_scheme.portfolio_history.copy()
        saved_reward_history = self.reward_scheme.reward_history.copy()
        saved_previous_value = self.reward_scheme.previous_value
        
        # 执行更多操作
        self.reward_scheme.calculate_reward(10300.0, -0.1, 103.0, {}, {}, 3)
        
        # 验证之前的状态已被保留并扩展
        assert len(self.reward_scheme.portfolio_history) == len(saved_portfolio_history) + 1
        assert self.reward_scheme.portfolio_history[:2] == saved_portfolio_history
        assert len(self.reward_scheme.reward_history) == len(saved_reward_history) + 1
        assert self.reward_scheme.reward_history[:2] == saved_reward_history